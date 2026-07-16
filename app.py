"""
Memorae Clone - Asistente de memoria personal por WhatsApp
=============================================================
Recibe mensajes de WhatsApp (via Twilio), usa Gemini para entender la intención
(recordatorio, nota, o pregunta general) y responde / guarda / programa avisos.
"""

import os
import json
import time
import base64
import threading
import sqlite3
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
DAILY_SUMMARY_HOUR = int(os.environ.get("DAILY_SUMMARY_HOUR", "8"))
WEATHER_LAT = os.environ.get("WEATHER_LAT", "-26.78")
WEATHER_LON = os.environ.get("WEATHER_LON", "-55.03")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
tz = ZoneInfo(APP_TIMEZONE)

app = Flask(__name__)

# --------------------------------------------------------------------------
# Base de datos (SQLite)
# --------------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT NOT NULL,
            content TEXT NOT NULL,
            due_at TEXT NOT NULL,
            sent INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS last_fired (
            phone TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            due_at TEXT NOT NULL,
            fired_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------
# Clasificación de intención con Gemini
# --------------------------------------------------------------------------
CLASSIFIER_SYSTEM_PROMPT = f"""Eres el motor de clasificación de un asistente de memoria por WhatsApp.
La fecha y hora actual es {{now}} (zona horaria {APP_TIMEZONE}).

Estos son los recordatorios PENDIENTES del usuario ahora mismo (id: contenido — fecha/hora):
{{pending_reminders}}

Dado un mensaje del usuario, responde SOLO con un JSON (sin texto adicional, sin markdown) con esta forma exacta:

{{{{
  "type": "reminder" | "note" | "list_reminders" | "list_notes" | "delete_reminders" | "delete_notes" | "delete_specific_reminder" | "edit_reminder" | "postpone" | "complete_reminder" | "help" | "stats" | "question",
  "content": "texto limpio del recordatorio o nota (null si no aplica)",
  "due_at": "fecha y hora en formato ISO 8601 con zona horaria del PRIMER (o único) aviso, o null si no aplica",
  "occurrences": ["fecha y hora ISO 8601 de cada aviso adicional"] o null si es un recordatorio de una sola vez,
  "target_ids": [lista de IDs (enteros) de la lista de recordatorios pendientes de arriba a los que se refiere el usuario, usando tu criterio semántico aunque las palabras no coincidan textualmente (ej. "ya me bañé" se refiere a un recordatorio "ir a bañarme"). Lista vacía si no aplica o no encontrás ninguno],
  "new_due_at": "para edit_reminder: la nueva fecha/hora en ISO 8601, o null si no aplica",
  "postpone_minutes": "para postpone: cuántos minutos posponer (número entero; si dice 'un rato' sin especificar, usa 10), o null si no aplica",
  "fact_to_remember": "un dato personal duradero sobre el usuario mencionado en el mensaje (nombre, gustos, trabajo, relaciones, preferencias, etc.), en pocas palabras y en tercera persona, o null si no hay ningún dato nuevo que valga la pena recordar"
}}}}

Reglas:
- "reminder": el usuario quiere que le avises algo en un momento futuro (ej. "recuérdame llamar al doctor mañana a las 5pm").
- "note": el usuario quiere guardar información sin fecha de aviso (ej. "anota que mi talla de zapato es 9").
- "list_reminders": el usuario pide ver sus recordatorios pendientes.
- "list_notes": el usuario pide ver sus notas guardadas.
- "delete_reminders": el usuario pide borrar/eliminar/limpiar TODOS sus recordatorios (ej. "elimina todos los recordatorios").
- "delete_notes": el usuario pide borrar/eliminar/limpiar TODAS sus notas guardadas.
- "delete_specific_reminder": el usuario pide borrar UN recordatorio puntual, no todos (ej. "borra el recordatorio de correr",
  "cancela el aviso de llamar al doctor"). Identificá cuál es usando la lista de pendientes de arriba y poné su id en "target_ids".
- "edit_reminder": el usuario pide cambiar la hora/fecha de un recordatorio existente (ej. "cambia el de correr para las 9am",
  "mueve el recordatorio del informe al viernes"). Poné el id en "target_ids" (solo uno), y en "new_due_at" la nueva fecha/hora.
- "postpone": el usuario pide posponer/aplazar/dejar para más tarde el recordatorio que acaba de sonar (ej. "posponer 10 min",
  "avisame en 15 minutos mejor", "dale, en un rato"). Pon el número de minutos en "postpone_minutes".
- "complete_reminder": el usuario avisa que ya hizo/completó algo pendiente, sin esperar a que suene el recordatorio
  (ej. "ya llamé al doctor", "listo, ya hice lo de correr", "ya me bañé" cuando hay un pendiente "ir a bañarme").
  Usá tu criterio semántico para identificar CUÁL de los pendientes de arriba corresponde, aunque las palabras no
  coincidan exactamente (conjugaciones distintas, sinónimos, etc.), y poné su id en "target_ids".
- "question": cualquier otra cosa, incluyendo preguntas generales tipo chat (incluye buscar en notas, ej. "¿qué anoté sobre el auto?").
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

Tenés acceso a Google Search en tiempo real. Para preguntas sobre noticias, resultados deportivos,
precios, clima, o cualquier cosa que pueda haber cambiado recientemente, buscá información actual
antes de responder en vez de adivinar. No digas que no tenés acceso a internet: si lo tenés.

Información importante sobre cómo funcionás en realidad (para responder bien si te preguntan):
- Cuando el usuario te pide un recordatorio, un sistema automático revisa cada minuto si ya llegó la hora,
  y en ese momento te manda un mensaje de WhatsApp SOLO, sin que el usuario tenga que pedirlo de nuevo.
- Si el recordatorio es recurrente (varios días), cada aviso ya quedó programado individualmente y se
  va a enviar solo en su fecha/hora correspondiente, sin necesidad de repetir el pedido.
- Si te preguntan si vas a avisar automáticamente, la respuesta es SÍ, siempre y cuando el recordatorio
  ya haya sido confirmado antes con un mensaje de "✅ Listo, te recordaré...".

Estas son las notas que el usuario te ha pedido guardar anteriormente. Úsalas para responder
si la pregunta se relaciona con algo que ya te contó (por ejemplo, "¿cuál es mi color favorito?"):
{notes}

Si la respuesta no está en las notas ni la sabes con certeza (ni buscando), dilo honestamente."""


def call_gemini(system_instruction: str, contents: list, max_retries: int = 3, use_search: bool = False) -> str:
    """Llama a la API REST de Gemini directamente (sin SDK pesado).
    Reintenta automáticamente si el servicio está saturado (503) o si se
    excedió el límite de solicitudes por minuto (429), con esperas más
    largas para el 429 (que necesita que se libere la cuota del minuto).
    Si use_search=True, activa "Grounding con Google Search" para preguntas
    que necesiten información actual (noticias, resultados deportivos, etc.)."""
    payload = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": contents,
        "generationConfig": {
            "thinkingConfig": {"thinkingBudget": 0},
            "maxOutputTokens": 500,
        },
    }
    if use_search:
        payload["tools"] = [{"google_search": {}}]

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


def describe_media(media_url: str, content_type: str) -> str:
    """Descarga un archivo multimedia de WhatsApp (via Twilio) y usa Gemini
    para transcribirlo (audio) o describir su contenido relevante (foto),
    devolviendo texto que se procesa igual que si el usuario lo hubiera escrito."""
    resp = requests.get(media_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=20)
    resp.raise_for_status()
    b64_data = base64.b64encode(resp.content).decode("utf-8")

    if content_type.startswith("audio"):
        prompt = (
            "Transcribe este audio de WhatsApp literalmente, en el idioma en que está hablado. "
            "Responde SOLO con la transcripción, sin comentarios adicionales."
        )
    elif content_type.startswith("image"):
        prompt = (
            "Describí brevemente el contenido relevante de esta imagen (texto visible, objetos "
            "importantes, contexto) para que un asistente personal lo use como si el usuario lo "
            "hubiera escrito. Responde SOLO con la descripción/texto extraído, sin comentarios adicionales."
        )
    else:
        raise ValueError(f"Tipo de archivo no soportado: {content_type}")

    contents = [{
        "role": "user",
        "parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": content_type, "data": b64_data}},
        ],
    }]
    return call_gemini("Eres un transcriptor/descriptor preciso y conciso.", contents).strip()


def classify_message(user_message: str, pending_reminders: list) -> dict:
    now_str = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    if pending_reminders:
        pending_text = "\n".join(
            f"- id {r['id']}: {r['content']} — {r['due_at']}" for r in pending_reminders
        )
    else:
        pending_text = "(el usuario no tiene recordatorios pendientes ahora mismo)"
    system = CLASSIFIER_SYSTEM_PROMPT.format(now=now_str, pending_reminders=pending_text)

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
    reply = call_gemini(system, contents, use_search=True).strip()

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
def process_and_reply(phone: str, incoming_msg: str, media_url: str = None, media_content_type: str = None):
    """Hace todo el trabajo pesado (clasificar con Gemini, guardar en BD) y
    manda la respuesta real por WhatsApp usando la API de Twilio directamente
    (no TwiML), para no depender del límite de tiempo del webhook."""
    if media_url and media_content_type:
        try:
            media_text = describe_media(media_url, media_content_type)
        except Exception as e:
            print(f"Error procesando archivo multimedia: {e}")
            send_whatsapp(phone, "⚠️ No pude procesar el audio/foto que mandaste. ¿Podés intentar de nuevo o escribirlo en texto?")
            return
        # Si además vino texto junto con el archivo (caption), lo combinamos
        incoming_msg = f"{incoming_msg}\n{media_text}".strip() if incoming_msg else media_text

    try:
        conn_lookup = get_db()
        pending_reminders = conn_lookup.execute(
            "SELECT id, content, due_at FROM reminders WHERE phone = ? AND sent = 0 ORDER BY due_at ASC",
            (phone,),
        ).fetchall()
        conn_lookup.close()
    except Exception as e:
        print(f"Error obteniendo recordatorios pendientes: {e}")
        pending_reminders = []

    try:
        result = classify_message(incoming_msg, pending_reminders)
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

        elif result["type"] == "delete_specific_reminder":
            ids = [i for i in (result.get("target_ids") or []) if isinstance(i, int)]
            if not ids:
                reply = "No identifiqué a cuál recordatorio te referís. Decime \"¿qué recordatorios tengo?\" para ver la lista completa."
            else:
                placeholders = ",".join("?" * len(ids))
                matches = conn.execute(
                    f"SELECT id, content, due_at FROM reminders WHERE phone = ? AND sent = 0 AND id IN ({placeholders})",
                    (phone, *ids),
                ).fetchall()
                if not matches:
                    reply = "No encontré ese recordatorio (puede que ya se haya borrado)."
                else:
                    conn.execute(
                        f"DELETE FROM reminders WHERE phone = ? AND id IN ({placeholders})",
                        (phone, *ids),
                    )
                    conn.commit()
                    if len(matches) == 1:
                        d = datetime.fromisoformat(matches[0]["due_at"])
                        reply = f"🗑️ Borré: \"{matches[0]['content']}\" — {d.strftime('%d/%m %H:%M')}"
                    else:
                        reply = f"🗑️ Borré {len(matches)} recordatorios: " + ", ".join(f'"{m["content"]}"' for m in matches)

        elif result["type"] == "edit_reminder":
            ids = [i for i in (result.get("target_ids") or []) if isinstance(i, int)]
            new_due_at = result.get("new_due_at")

            if not new_due_at:
                reply = "No entendí bien a qué hora querés moverlo. ¿Podés decirlo de nuevo con la fecha/hora exacta?"
            elif not ids:
                reply = "No identifiqué a cuál recordatorio te referís. Decime \"¿qué recordatorios tengo?\" para ver la lista completa."
            elif len(ids) > 1:
                reply = "Encontré más de un recordatorio que podría coincidir — decime cuál más específicamente."
            else:
                match = conn.execute(
                    "SELECT content FROM reminders WHERE phone = ? AND sent = 0 AND id = ?",
                    (phone, ids[0]),
                ).fetchone()
                if not match:
                    reply = "No encontré ese recordatorio (puede que ya se haya borrado)."
                else:
                    conn.execute("UPDATE reminders SET due_at = ? WHERE id = ?", (new_due_at, ids[0]))
                    conn.commit()
                    d = datetime.fromisoformat(new_due_at)
                    reply = f"✅ Listo, moví \"{match['content']}\" para el {d.strftime('%d/%m/%Y %H:%M')}"

        elif result["type"] == "postpone":
            last = conn.execute(
                "SELECT content FROM last_fired WHERE phone = ?", (phone,)
            ).fetchone()
            minutes = result.get("postpone_minutes") or 10
            if not last:
                reply = "No tengo ningún recordatorio reciente para posponer. ¿Cuál querés que te vuelva a avisar?"
            else:
                new_due = datetime.now(tz) + timedelta(minutes=minutes)
                conn.execute(
                    "INSERT INTO reminders (phone, content, due_at, created_at) VALUES (?, ?, ?, ?)",
                    (phone, last["content"], new_due.isoformat(), now_iso),
                )
                conn.commit()
                reply = f"⏳ Listo, te lo vuelvo a recordar en {minutes} minutos: \"{last['content']}\""

        elif result["type"] == "complete_reminder":
            ids = [i for i in (result.get("target_ids") or []) if isinstance(i, int)]
            if not ids:
                reply = "No identifiqué a cuál recordatorio te referís. Decime \"¿qué recordatorios tengo?\" para ver la lista completa."
            else:
                placeholders = ",".join("?" * len(ids))
                matches = conn.execute(
                    f"SELECT id, content FROM reminders WHERE phone = ? AND sent = 0 AND id IN ({placeholders})",
                    (phone, *ids),
                ).fetchall()
                if not matches:
                    reply = "No encontré ese recordatorio (puede que ya esté marcado como hecho)."
                else:
                    conn.execute(
                        f"UPDATE reminders SET sent = 1 WHERE phone = ? AND id IN ({placeholders})",
                        (phone, *ids),
                    )
                    conn.commit()
                    if len(matches) == 1:
                        reply = f"✅ ¡Buenísimo! Marqué como hecho: \"{matches[0]['content']}\""
                    else:
                        reply = f"✅ Marqué como hechos: " + ", ".join(f'"{m["content"]}"' for m in matches)

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
    num_media = int(request.values.get("NumMedia", "0") or "0")
    media_url = request.values.get("MediaUrl0") if num_media > 0 else None
    media_content_type = request.values.get("MediaContentType0") if num_media > 0 else None

    # Respondemos a Twilio al instante (sin esperar a Gemini), para no
    # depender de su límite de tiempo. El mensaje real se manda aparte,
    # en un hilo en segundo plano, usando la API de Twilio directamente.
    resp = MessagingResponse()

    if not incoming_msg and not media_url:
        resp.message("No recibí ningún texto ni archivo. ¿Puedes intentar de nuevo?")
        return Response(str(resp), mimetype="application/xml")

    threading.Thread(
        target=process_and_reply,
        args=(phone, incoming_msg, media_url, media_content_type),
        daemon=True,
    ).start()
    return Response(str(resp), mimetype="application/xml")


# --------------------------------------------------------------------------
# Envío proactivo de recordatorios vencidos
# --------------------------------------------------------------------------
def check_due_reminders():
    conn = get_db()
    now_iso = datetime.now(tz).isoformat()
    rows = conn.execute(
        "SELECT id, phone, content, due_at FROM reminders WHERE sent = 0 AND due_at <= ?",
        (now_iso,),
    ).fetchall()

    for r in rows:
        try:
            twilio_client.messages.create(
                from_=TWILIO_WHATSAPP_NUMBER,
                to=r["phone"],
                body=f"⏰ Recordatorio: {r['content']}\n(si querés, respondé \"posponer 10 min\")",
            )
            conn.execute("UPDATE reminders SET sent = 1 WHERE id = ?", (r["id"],))
            conn.execute("DELETE FROM last_fired WHERE phone = ?", (r["phone"],))
            conn.execute(
                "INSERT INTO last_fired (phone, content, due_at, fired_at) VALUES (?, ?, ?, ?)",
                (r["phone"], r["content"], r["due_at"], now_iso),
            )
            conn.commit()
        except Exception as e:
            print(f"Error enviando recordatorio {r['id']}: {e}")

    conn.close()


WEATHER_CODE_DESCRIPTIONS = {
    0: ("☀️", "despejado"), 1: ("🌤️", "mayormente despejado"), 2: ("⛅", "parcialmente nublado"),
    3: ("☁️", "nublado"), 45: ("🌫️", "neblina"), 48: ("🌫️", "neblina con escarcha"),
    51: ("🌦️", "llovizna ligera"), 53: ("🌦️", "llovizna"), 55: ("🌧️", "llovizna densa"),
    61: ("🌧️", "lluvia ligera"), 63: ("🌧️", "lluvia"), 65: ("🌧️", "lluvia fuerte"),
    71: ("🌨️", "nieve ligera"), 73: ("🌨️", "nieve"), 75: ("❄️", "nieve fuerte"),
    80: ("🌦️", "chubascos ligeros"), 81: ("🌧️", "chubascos"), 82: ("⛈️", "chubascos fuertes"),
    95: ("⛈️", "tormenta"), 96: ("⛈️", "tormenta con granizo"), 99: ("⛈️", "tormenta fuerte con granizo"),
}


def get_weather_summary() -> str | None:
    """Consulta el clima actual con Open-Meteo (gratis, sin API key)."""
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": WEATHER_LAT,
                "longitude": WEATHER_LON,
                "current": "temperature_2m,weather_code",
                "timezone": "auto",
            },
            timeout=10,
        )
        resp.raise_for_status()
        current = resp.json()["current"]
        temp = round(current["temperature_2m"])
        emoji, desc = WEATHER_CODE_DESCRIPTIONS.get(current["weather_code"], ("🌡️", "condiciones variables"))
        return f"{emoji} {temp}°C, {desc}"
    except Exception as e:
        print(f"Error consultando el clima: {e}")
        return None


def send_daily_summaries():
    """Cada mañana (a la hora configurada), le manda a cada usuario con
    recordatorios pendientes para hoy un resumen único con todo lo que tiene."""
    conn = get_db()
    today_str = datetime.now(tz).strftime("%Y-%m-%d")
    phones = conn.execute(
        "SELECT DISTINCT phone FROM reminders WHERE sent = 0"
    ).fetchall()

    weather = get_weather_summary()

    for row in phones:
        phone = row["phone"]
        todays = conn.execute(
            "SELECT content, due_at FROM reminders WHERE phone = ? AND sent = 0 AND due_at LIKE ? ORDER BY due_at ASC",
            (phone, f"{today_str}%"),
        ).fetchall()
        if not todays:
            continue
        lines = ["☀️ ¡Buenos días! Esto es lo que tenés para hoy:"]
        if weather:
            lines.append(f"Clima: {weather}")
            lines.append("")
        for r in todays:
            d = datetime.fromisoformat(r["due_at"])
            lines.append(f"• {r['content']} — {d.strftime('%H:%M')}")
        send_whatsapp(phone, "\n".join(lines))

    conn.close()


scheduler = BackgroundScheduler(timezone=str(tz))
scheduler.add_job(check_due_reminders, "interval", minutes=1)
scheduler.add_job(send_daily_summaries, "cron", hour=DAILY_SUMMARY_HOUR, minute=0)
scheduler.start()


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "time": datetime.now(tz).isoformat()}


DASHBOARD_TOKEN = os.environ.get("DASHBOARD_TOKEN", "")

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mis recordatorios y notas</title>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; background: #f4f4f5; margin: 0; padding: 24px; color: #1a1a1a; }
  h1 { font-size: 1.4rem; margin-bottom: 4px; }
  .subtitle { color: #666; margin-bottom: 24px; font-size: 0.9rem; }
  .card { background: white; border-radius: 12px; padding: 16px 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  .card h2 { font-size: 1.05rem; margin: 0 0 12px 0; }
  ul { list-style: none; padding: 0; margin: 0; }
  li { padding: 10px 0; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; gap: 12px; }
  li:last-child { border-bottom: none; }
  .content { flex: 1; }
  .meta { color: #888; font-size: 0.85rem; white-space: nowrap; }
  .empty { color: #999; font-style: italic; padding: 8px 0; }
  select { margin-bottom: 16px; padding: 6px 10px; border-radius: 8px; border: 1px solid #ddd; }
</style>
</head>
<body>
  <h1>📱 Memorae Pro</h1>
  <div class="subtitle">Actualizado: {{ now }}</div>

  <div class="card">
    <h2>📌 Recordatorios pendientes ({{ reminders|length }})</h2>
    {% if reminders %}
    <ul>
      {% for r in reminders %}
      <li><span class="content">{{ r.content }}</span><span class="meta">{{ r.due_at }}</span></li>
      {% endfor %}
    </ul>
    {% else %}
    <div class="empty">No tenés recordatorios pendientes.</div>
    {% endif %}
  </div>

  <div class="card">
    <h2>🗒️ Notas guardadas ({{ notes|length }})</h2>
    {% if notes %}
    <ul>
      {% for n in notes %}
      <li><span class="content">{{ n.content }}</span><span class="meta">{{ n.created_at }}</span></li>
      {% endfor %}
    </ul>
    {% else %}
    <div class="empty">No tenés notas guardadas.</div>
    {% endif %}
  </div>
</body>
</html>"""


@app.route("/dashboard", methods=["GET"])
def dashboard():
    if not DASHBOARD_TOKEN or request.args.get("token") != DASHBOARD_TOKEN:
        return "No autorizado. Falta el token correcto en la URL (?token=...).", 401

    phone = request.args.get("phone", "")
    conn = get_db()

    if phone:
        reminders = conn.execute(
            "SELECT content, due_at FROM reminders WHERE phone = ? AND sent = 0 ORDER BY due_at ASC",
            (phone,),
        ).fetchall()
        notes = conn.execute(
            "SELECT content, created_at FROM notes WHERE phone = ? ORDER BY id DESC",
            (phone,),
        ).fetchall()
    else:
        reminders = conn.execute(
            "SELECT content, due_at FROM reminders WHERE sent = 0 ORDER BY due_at ASC"
        ).fetchall()
        notes = conn.execute(
            "SELECT content, created_at FROM notes ORDER BY id DESC"
        ).fetchall()
    conn.close()

    from flask import render_template_string
    return render_template_string(
        DASHBOARD_HTML,
        reminders=reminders,
        notes=notes,
        now=datetime.now(tz).strftime("%d/%m/%Y %H:%M"),
    )


if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
else:
    init_db()
