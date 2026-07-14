"""
Memorae Clone - Asistente de memoria personal por WhatsApp
=============================================================
Recibe mensajes de WhatsApp (via Twilio), usa Gemini para entender la intención
(recordatorio, nota, o pregunta general) y responde / guarda / programa avisos.
"""

import os
import json
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
TWILIO_WHATSAPP_NUMBER = os.environ["TWILIO_WHATSAPP_NUMBER"] # ej: whatsapp:+14155238886
APP_TIMEZONE = os.environ.get("APP_TIMEZONE", "America/Mexico_City")
DB_PATH = os.environ.get("DB_PATH", "memorae.db")
GEMINI_MODEL = "gemini-3.5-flash"
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
  "type": "reminder" | "note" | "list_reminders" | "list_notes" | "question",
  "content": "texto limpio del recordatorio o nota (null si no aplica)",
  "due_at": "fecha y hora en formato ISO 8601 con zona horaria, o null si no es un recordatorio o no se especificó hora"
}}}}

Reglas:
- "reminder": el usuario quiere que le avises algo en un momento futuro (ej. "recuérdame llamar al doctor mañana a las 5pm").
- "note": el usuario quiere guardar información sin fecha de aviso (ej. "anota que mi talla de zapato es 9").
- "list_reminders": el usuario pide ver sus recordatorios pendientes.
- "list_notes": el usuario pide ver sus notas guardadas.
- "question": cualquier otra cosa, incluyendo preguntas generales tipo chat.
- Si el usuario da una hora relativa ("en 2 horas", "mañana", "el viernes"), calcula la fecha absoluta usando la fecha/hora actual dada arriba.
- Si es "reminder" pero no dio ninguna indicación de tiempo, trátalo como "note" en vez de "reminder".
"""

CHAT_SYSTEM_PROMPT = """Eres un asistente personal amigable y conciso que vive dentro de WhatsApp.
Respondes preguntas generales con claridad y brevedad, en el mismo idioma en que te escriben.
Si no sabes algo con certeza, dilo honestamente."""


def call_gemini(system_instruction: str, contents: list) -> str:
    """Llama a la API REST de Gemini directamente (sin SDK pesado)."""
    payload = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": contents,
        # Desactivamos el modo "thinking": para clasificar mensajes cortos
        # no lo necesitamos, y nos ahorra varios segundos de latencia.
        "generationConfig": {
            "thinkingConfig": {"thinkingBudget": 0},
            "maxOutputTokens": 500,
        },
    }
    resp = requests.post(
        GEMINI_API_URL,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY,
        },
        json=payload,
        timeout=55,
    )
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
        "SELECT role, content FROM conversation_history WHERE phone = ? ORDER BY id DESC LIMIT 10",
        (phone,),
    ).fetchall()
    conn.close()

    # Gemini usa "model" en vez de "assistant" como rol, y "parts" en vez de "content"
    contents = [
        {"role": "model" if r["role"] == "assistant" else "user", "parts": [{"text": r["content"]}]}
        for r in reversed(history_rows)
    ]
    contents.append({"role": "user", "parts": [{"text": user_message}]})

    reply = call_gemini(CHAT_SYSTEM_PROMPT, contents).strip()

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
@app.route("/webhook", methods=["POST"])
def webhook():
    incoming_msg = request.values.get("Body", "").strip()
    phone = request.values.get("From", "") # ej: whatsapp:+52155...

    resp = MessagingResponse()
    msg = resp.message()

    if not incoming_msg:
        msg.body("No recibí ningún texto. ¿Puedes intentar de nuevo?")
        return Response(str(resp), mimetype="application/xml")

    result = classify_message(incoming_msg)
    conn = get_db()
    now_iso = datetime.now(tz).isoformat()

    if result["type"] == "reminder" and result.get("due_at"):
        conn.execute(
            "INSERT INTO reminders (phone, content, due_at, created_at) VALUES (?, ?, ?, ?)",
            (phone, result["content"], result["due_at"], now_iso),
        )
        conn.commit()
        due_dt = datetime.fromisoformat(result["due_at"])
        msg.body(f"✅ Listo, te recordaré: \"{result['content']}\"\n🕒 {due_dt.strftime('%d/%m/%Y %H:%M')}")

    elif result["type"] == "note":
        conn.execute(
            "INSERT INTO notes (phone, content, created_at) VALUES (?, ?, ?)",
            (phone, result["content"], now_iso),
        )
        conn.commit()
        msg.body(f"📝 Anotado: \"{result['content']}\"")

    elif result["type"] == "list_reminders":
        rows = conn.execute(
            "SELECT content, due_at FROM reminders WHERE phone = ? AND sent = 0 ORDER BY due_at ASC",
            (phone,),
        ).fetchall()
        if not rows:
            msg.body("No tienes recordatorios pendientes.")
        else:
            lines = ["📌 Tus recordatorios pendientes:"]
            for r in rows:
                d = datetime.fromisoformat(r["due_at"])
                lines.append(f"• {r['content']} — {d.strftime('%d/%m %H:%M')}")
            msg.body("\n".join(lines))

    elif result["type"] == "list_notes":
        rows = conn.execute(
            "SELECT content, created_at FROM notes WHERE phone = ? ORDER BY id DESC LIMIT 20",
            (phone,),
        ).fetchall()
        if not rows:
            msg.body("Todavía no tienes notas guardadas.")
        else:
            lines = ["🗒️ Tus notas:"]
            for r in rows:
                lines.append(f"• {r['content']}")
            msg.body("\n".join(lines))

    else: # question
        conn.close()
        reply = answer_general_question(phone, incoming_msg)
        msg.body(reply)
        return Response(str(resp), mimetype="application/xml")

    conn.close()
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
