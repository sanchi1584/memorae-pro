"""
Memorae Clone - Asistente de memoria personal por WhatsApp
=============================================================
Recibe mensajes de WhatsApp (via Twilio), usa Gemini para entender la intención
(recordatorio, nota, o pregunta general) y responde / guarda / programa avisos.
"""

import os
import json
import sqlite3
import time
import threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from flask import Flask, request, Response
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioClient
from apscheduler.schedulers.background import BackgroundScheduler
import requests
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
TWILIO_WHATSAPP_NUMBER = os.environ["TWILIO_WHATSAPP_NUMBER"]  # ej: whatsapp:+14155238886
APP_TIMEZONE = os.environ.get("APP_TIMEZONE", "America/Mexico_City")
DB_PATH = os.environ.get("DB_PATH", "memorae.db")
GEMINI_MODEL = "gemini-flash-lite-latest"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
tz = ZoneInfo(APP_TIMEZONE)

app = Flask(__name__)

# --------------------------------------------------------------------------
# Base de datos
# --------------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT NOT NULL,
            content TEXT NOT NULL,
            due_at TEXT NOT NULL,
            sent INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------
# Clasificación de intención con Gemini
# --------------------------------------------------------------------------
CLASSIFIER_SYSTEM_PROMPT = f"""Eres el motor de clasificación de un asistente de memoria por WhatsApp.
La fecha y hora actual es {{now}} (zona horaria {APP_TIMEZONE}).

Dado un mensaje del usuario, responde SOLO con un JSON (sin texto adicional, sin markdown) con esta forma exacta:

{{{{
  "type": "reminder" | "note" | "list_reminders" | "list_notes" | "delete_reminders" | "delete_notes" | "question",
  "content": "texto limpio del recordatorio o nota (null si no aplica)",
  "due_at": "fecha y hora en formato ISO 8601 con zona horaria del PRIMER (o único) aviso, o null si no aplica",
  "occurrences": ["fecha y hora ISO 8601 de cada aviso adicional"] o null si es un recordatorio de una sola vez,
  "fact_to_remember": "un dato personal duradero sobre el usuario mencionado en el mensaje (nombre, gustos, trabajo, relaciones, preferencias, etc.), en pocas palabras y en tercera persona, o null si no hay ningún dato nuevo que valga la pena recordar"
}}}}

Reglas:
- "reminder": el usuario quiere que le avises algo en un momento futuro (ej. "recuérdame llamar al doctor mañana a las 5pm").
- "note": el usuario quiere guardar información sin fecha de aviso (ej. "anota que mi talla de zapato es 9").
- "list_reminders": el usuario pide ver sus recordatorios pendientes.
- "list_notes": el usuario pide ver sus notas guardadas.
- "delete_reminders": el usuario pide borrar/eliminar/limpiar TODOS sus recordatorios (ej. "elimina todos los recordatorios", "borra mis recordatorios").
- "delete_notes": el usuario pide borrar/eliminar/limpiar TODAS sus notas guardadas.
- "question": cualquier otra cosa, incluyendo preguntas generales tipo chat.
- Si el usuario da una hora relativa ("en 2 horas", "mañana", "el viernes"), calcula la fecha absoluta usando la fecha/hora actual dada arriba.
- Si es "reminder" pero no dio ninguna indicación de tiempo, trátalo como "note" en vez de "reminder".
- RECORDATORIOS RECURRENTES: si el usuario pide que se repita ("todos los días", "de lunes a viernes",
  "hasta el viernes", "cada día a partir de mañana"), calcula TODAS las fechas/horas concretas dentro del
  rango que haya indicado (excluyendo fines de semana solo si el usuario lo pidió explícitamente) y ponlas
  en "occurrences", SIN REPETIR NINGUNA FECHA (una sola vez por día). La primera fecha va en "due_at" y el
  resto (una por cada día restante) en "occurrences". Si el usuario no puso una fecha final ("todos los
  días" sin decir hasta cuándo), asume un límite razonable de 14 días desde hoy. Nunca generes más de 30
  fechas en total, y nunca repitas la misma fecha/hora dos veces.
- "fact_to_remember" se aplica SIN IMPORTAR el "type": aunque el mensaje sea una pregunta o un comentario casual,
  si menciona algo duradero sobre el usuario (ej. "me llamo Santiago", "vivo en Buenos Aires", "no me gusta el picante"),
  extráelo aquí. Si el mensaje no aporta ningún dato nuevo sobre el usuario, deja este campo en null.
"""



CHAT_SYSTEM_PROMPT = """Eres un asistente personal amigable y conciso que vive dentro de WhatsApp.
La fecha y hora actual es {now} (zona horaria {tzname}).
Respondes preguntas generales con claridad y brevedad, en el mismo idioma en que te escriben.
Si te preguntan la hora, la fecha, o el día de la semana, respóndelo directamente usando el dato de arriba.

Estas son las notas que el usuario te ha pedido guardar anteriormente. Úsalas para responder
si la pregunta se relaciona con algo que ya te contó (por ejemplo, "¿cuál es mi color favorito?"):
{notes}

Si la respuesta no está en las notas ni la sabes con certeza, dilo honestamente."""


def call_gemini(system_instruction: str, contents: list, max_retries: int = 3) -> str:
    """Llama a la API REST de Gemini directamente (sin SDK pesado).
    Reintenta automáticamente si el servicio está saturado (503) o si se
    excedió el límite de solicitudes por minuto (429), con esperas más
    largas para el 429 (que necesita que se libere la cuota del minuto)."""
    payload = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": contents,
        "generationConfig": {
            "thinkingConfig": {"thinkingBudget": 0},
            "maxOutputTokens": 500,
        },
    }

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                GEMINI_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": GEMINI_API_KEY,
                },
                json=payload,
                timeout=20,
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(3 + attempt * 3)
                continue
            raise

        if resp.status_code == 429 and attempt < max_retries:
            wait = 15 + attempt * 10  # el límite de RPM tarda en liberarse
            print(f"Gemini devolvió 429 (límite de cuota), esperando {wait}s antes de reintentar (intento {attempt + 1}/{max_retries + 1})...")
            time.sleep(wait)
            continue
        if resp.status_code == 503 and attempt < max_retries:
            wait = 2 + attempt * 3
            print(f"Gemini devolvió 503 (saturado), esperando {wait}s antes de reintentar (intento {attempt + 1}/{max_retries + 1})...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


def classify_message(user_message: str) -> dict:
    now_str = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    system = CLASSIFIER_SYSTEM_PROMPT.format(now=now_str)

    contents = [{"role": "user", "parts": [{"text": user_message}]}]
    raw = call_gemini(system, contents).strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Si el modelo no devuelve JSON válido, lo tratamos como pregunta general
        return {"type": "question", "content": user_message, "due_at": None}


def answer_general_question(phone: str, user_message: str) -> str:
    conn = get_db()
    history_rows = conn.execute(
        "SELECT role, content FROM conversation_history WHERE phone = ? ORDER BY id DESC LIMIT 30",
        (phone,),
    ).fetchall()
    note_rows = conn.execute(
        "SELECT content FROM notes WHERE phone = ? ORDER BY id DESC LIMIT 50",
        (phone,),
    ).fetchall()
    conn.close()

    # Gemini usa "model" en vez de "assistant" como rol, y "parts" en vez de "content"
    contents = [
        {"role": "model" if r["role"] == "assistant" else "user", "parts": [{"text": r["content"]}]}
        for r in reversed(history_rows)
    ]
    contents.append({"role": "user", "parts": [{"text": user_message}]})

    now_str = datetime.now(tz).strftime("%A %d de %B de %Y, %H:%M")
    notes_text = "\n".join(f"- {r['content']}" for r in note_rows) if note_rows else "(el usuario no tiene notas guardadas todavía)"
    system = CHAT_SYSTEM_PROMPT.format(now=now_str, tzname=APP_TIMEZONE, notes=notes_text)
    reply = call_gemini(system, contents).strip()

    conn = get_db()
    now_iso = datetime.now(tz).isoformat()
    conn.execute(
        "INSERT INTO conversation_history (phone, role, content, created_at) VALUES (?, ?, ?, ?)",
        (phone, "user", user_message, now_iso),
    )
    conn.execute(
        "INSERT INTO conversation_history (phone, role, content, created_at) VALUES (?, ?, ?, ?)",
        (phone, "assistant", reply, now_iso),
    )
    conn.commit()
    conn.close()
    return reply


# --------------------------------------------------------------------------
# Webhook de Twilio
# --------------------------------------------------------------------------
def process_and_reply(phone: str, incoming_msg: str):
    """Hace todo el trabajo pesado (clasificar con Gemini, guardar en BD) y
    manda la respuesta real por WhatsApp usando la API de Twilio directamente
    (no TwiML), para no depender del límite de tiempo del webhook."""
    try:
        result = classify_message(incoming_msg)
    except Exception as e:
        print(f"Error clasificando mensaje: {e}")
        send_whatsapp(phone, "⚠️ Tuve un problema entendiendo tu mensaje (el servicio de IA no respondió). Intenta de nuevo en un momento.")
        return

    # Protección: si el modelo dijo "reminder" o "note" pero no vino contenido
    # (puede pasar de vez en cuando), lo tratamos como pregunta general en vez
    # de fallar al intentar guardar una nota/recordatorio vacío.
    if result.get("type") in ("reminder", "note") and not (result.get("content") or "").strip():
        print(f"Advertencia: '{result.get('type')}' sin contenido, tratando como pregunta general.")
        result["type"] = "question"

    conn = get_db()
    now_iso = datetime.now(tz).isoformat()

    try:
        # Si el clasificador detectó un dato personal duradero (aunque el mensaje
        # sea una pregunta o comentario casual), lo guardamos como nota aparte,
        # sin duplicar el "content" si ya se guardó como nota explícita.
        fact = result.get("fact_to_remember")
        if fact and not (result["type"] == "note" and fact.strip() == (result.get("content") or "").strip()):
            conn.execute(
                "INSERT INTO notes (phone, content, created_at) VALUES (?, ?, ?)",
                (phone, fact, now_iso),
            )
            conn.commit()

        if result["type"] == "reminder" and result.get("due_at"):
            all_dates = [result["due_at"]] + [d for d in (result.get("occurrences") or []) if d]
            # Quitamos fechas duplicadas (por si el modelo repitió alguna) y limitamos a 30
            seen = set()
            unique_dates = []
            for d in all_dates:
                if d not in seen:
                    seen.add(d)
                    unique_dates.append(d)
            all_dates = unique_dates[:30]

            inserted = 0
            for due_at in all_dates:
                exists = conn.execute(
                    "SELECT 1 FROM reminders WHERE phone = ? AND content = ? AND due_at = ? AND sent = 0",
                    (phone, result["content"], due_at),
                ).fetchone()
                if not exists:
                    conn.execute(
                        "INSERT INTO reminders (phone, content, due_at, created_at) VALUES (?, ?, ?, ?)",
                        (phone, result["content"], due_at, now_iso),
                    )
                    inserted += 1
            conn.commit()
            first_dt = datetime.fromisoformat(all_dates[0])
            if len(all_dates) > 1:
                last_dt = datetime.fromisoformat(all_dates[-1])
                reply = (
                    f"✅ Listo, te recordaré: \"{result['content']}\"\n"
                    f"🔁 {len(all_dates)} avisos, desde el {first_dt.strftime('%d/%m/%Y %H:%M')} "
                    f"hasta el {last_dt.strftime('%d/%m/%Y %H:%M')}"
                )
            else:
                reply = f"✅ Listo, te recordaré: \"{result['content']}\"\n🕒 {first_dt.strftime('%d/%m/%Y %H:%M')}"

        elif result["type"] == "note":
            conn.execute(
                "INSERT INTO notes (phone, content, created_at) VALUES (?, ?, ?)",
                (phone, result["content"], now_iso),
            )
            conn.commit()
            reply = f"📝 Anotado: \"{result['content']}\""

        elif result["type"] == "list_reminders":
            rows = conn.execute(
                "SELECT content, due_at FROM reminders WHERE phone = ? AND sent = 0 ORDER BY due_at ASC",
                (phone,),
            ).fetchall()
            if not rows:
                reply = "No tienes recordatorios pendientes."
            else:
                lines = ["📌 Tus recordatorios pendientes:"]
                for r in rows:
                    d = datetime.fromisoformat(r["due_at"])
                    lines.append(f"• {r['content']} — {d.strftime('%d/%m %H:%M')}")
                reply = "\n".join(lines)

        elif result["type"] == "list_notes":
            rows = conn.execute(
                "SELECT content, created_at FROM notes WHERE phone = ? ORDER BY id DESC LIMIT 20",
                (phone,),
            ).fetchall()
            if not rows:
                reply = "Todavía no tienes notas guardadas."
            else:
                lines = ["🗒️ Tus notas:"]
                for r in rows:
                    lines.append(f"• {r['content']}")
                reply = "\n".join(lines)

        elif result["type"] == "delete_reminders":
            cur = conn.execute("DELETE FROM reminders WHERE phone = ?", (phone,))
            conn.commit()
            reply = f"🗑️ Listo, borré {cur.rowcount} recordatorio(s)." if cur.rowcount else "No tenías recordatorios para borrar."

        elif result["type"] == "delete_notes":
            cur = conn.execute("DELETE FROM notes WHERE phone = ?", (phone,))
            conn.commit()
            reply = f"🗑️ Listo, borré {cur.rowcount} nota(s)." if cur.rowcount else "No tenías notas para borrar."

        else:  # question
            try:
                reply = answer_general_question(phone, incoming_msg)
            except Exception as e:
                print(f"Error respondiendo pregunta general: {e}")
                reply = "⚠️ Tuve un problema pensando la respuesta (el servicio de IA no respondió). Intenta de nuevo en un momento."
    except Exception as e:
        print(f"Error inesperado procesando el mensaje: {e}")
        reply = "⚠️ Tuve un problema procesando tu mensaje. Intenta de nuevo en un momento."
    finally:
        conn.close()

    send_whatsapp(phone, reply)


def send_whatsapp(phone: str, body: str):
    """Manda un mensaje de WhatsApp usando la API de Twilio (no TwiML)."""
    try:
        twilio_client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            to=phone,
            body=body,
        )
    except Exception as e:
        print(f"Error mandando mensaje de WhatsApp a {phone}: {e}")


@app.route("/webhook", methods=["POST"])
def webhook():
    incoming_msg = request.values.get("Body", "").strip()
    phone = request.values.get("From", "")  # ej: whatsapp:+52155...

    # Respondemos a Twilio al instante (sin esperar a Gemini), para no
    # depender de su límite de tiempo. El mensaje real se manda aparte,
    # en un hilo en segundo plano, usando la API de Twilio directamente.
    resp = MessagingResponse()

    if not incoming_msg:
        resp.message("No recibí ningún texto. ¿Puedes intentar de nuevo?")
        return Response(str(resp), mimetype="application/xml")

    threading.Thread(target=process_and_reply, args=(phone, incoming_msg), daemon=True).start()
    return Response(str(resp), mimetype="application/xml")


# --------------------------------------------------------------------------
# Envío proactivo de recordatorios vencidos
# --------------------------------------------------------------------------
def check_due_reminders():
    conn = get_db()
    now_iso = datetime.now(tz).isoformat()
    rows = conn.execute(
        "SELECT id, phone, content FROM reminders WHERE sent = 0 AND due_at <= ?",
        (now_iso,),
    ).fetchall()

    for r in rows:
        try:
            twilio_client.messages.create(
                from_=TWILIO_WHATSAPP_NUMBER,
                to=r["phone"],
                body=f"⏰ Recordatorio: {r['content']}",
            )
            conn.execute("UPDATE reminders SET sent = 1 WHERE id = ?", (r["id"],))
        except Exception as e:
            print(f"Error enviando recordatorio {r['id']}: {e}")

    conn.commit()
    conn.close()


scheduler = BackgroundScheduler(timezone=str(tz))
scheduler.add_job(check_due_reminders, "interval", minutes=1)
scheduler.start()


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "time": datetime.now(tz).isoformat()}


if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
else:
    init_db()
