"""
Memorae Clone - Asistente de memoria personal por WhatsApp
=============================================================
Recibe mensajes de WhatsApp (via Twilio o via la API de Meta/Cloud API,
según WHATSAPP_PROVIDER), usa Gemini para entender la intención
(recordatorio, nota, o pregunta general) y responde / guarda / programa avisos.

Rutas de webhook:
- /webhook       -> Twilio (form-encoded)
- /webhook-meta  -> Meta / WhatsApp Cloud API (JSON), con verificación GET
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

# Proveedor de WhatsApp activo: "twilio" o "meta". Se cambia con una variable
# de entorno en Render, sin tocar código, para poder volver atrás si algo falla.
WHATSAPP_PROVIDER = os.environ.get("WHATSAPP_PROVIDER", "twilio").strip().lower()

# --- Twilio (opcional si WHATSAPP_PROVIDER == "meta") ---
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_NUMBER = os.environ.get("TWILIO_WHATSAPP_NUMBER", "")  # ej: whatsapp:+14155238886

# --- Meta / WhatsApp Cloud API (opcional si WHATSAPP_PROVIDER == "twilio") ---
META_PHONE_NUMBER_ID = os.environ.get("META_PHONE_NUMBER_ID", "")
META_ACCESS_TOKEN = os.environ.get("META_ACCESS_TOKEN", "")
META_VERIFY_TOKEN = os.environ.get("META_VERIFY_TOKEN", "memorae_verify")
META_API_VERSION = os.environ.get("META_API_VERSION", "v20.0")
META_MESSAGES_URL = f"https://graph.facebook.com/{META_API_VERSION}/{META_PHONE_NUMBER_ID}/messages"

APP_TIMEZONE = os.environ.get("APP_TIMEZONE", "America/Mexico_City")
DB_PATH = os.environ.get("DB_PATH", "memorae.db")
GEMINI_MODEL = "gemini-flash-lite-latest"
DAILY_SUMMARY_HOUR = int(os.environ.get("DAILY_SUMMARY_HOUR", "6"))
DAILY_SUMMARY_MINUTE = int(os.environ.get("DAILY_SUMMARY_MINUTE", "30"))
WEATHER_LAT = os.environ.get("WEATHER_LAT", "-26.7825")
WEATHER_LON = os.environ.get("WEATHER_LON", "-55.0339")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None
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

No tenés acceso a internet en tiempo real. Para preguntas sobre noticias, resultados deportivos,
precios actuales, o cualquier cosa que pueda haber cambiado recientemente, aclará que no tenés
información actualizada al respecto en vez de inventar una respuesta.

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


def get_meta_media_url(media_id: str) -> str | None:
    """Meta no manda la URL del archivo directamente en el webhook, solo un
    media_id. Hay que pedirle a la Graph API la URL real (válida ~5 minutos)."""
    try:
        resp = requests.get(
            f"https://graph.facebook.com/{META_API_VERSION}/{media_id}",
            headers={"Authorization": f"Bearer {META_ACCESS_TOKEN}"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("url")
    except Exception as e:
        print(f"Error obteniendo URL de media de Meta ({media_id}): {e}")
        return None


def describe_media(media_url: str, content_type: str) -> str:
    """Descarga un archivo multimedia de WhatsApp (via Twilio) y usa Gemini
    para transcribirlo (audio) o describir su contenido relevante (foto),
    devolviendo texto que se procesa igual que si el usuario lo hubiera escrito."""
    if WHATSAPP_PROVIDER == "meta":
        resp = requests.get(media_url, headers={"Authorization": f"Bearer {META_ACCESS_TOKEN}"}, timeout=20)
    else:
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
def process_and_reply(phone: str, incoming_msg: str, media_url: str = None, media_content_type: str = None):
    """Hace todo el trabajo pesado (clasificar con Gemini, guardar en BD) y
    manda la respuesta real por WhatsApp usando la API de Twilio directamente
    (no TwiML), para no depender del límite de tiempo del webhook."""
    print(f"[process_and_reply] INICIO — phone={phone!r} msg={incoming_msg!r} media_url={bool(media_url)}")
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
 