# Memorae Clone — Asistente de memoria por WhatsApp

Bot que recibe mensajes de WhatsApp, entiende si es un **recordatorio**, una **nota** o una
**pregunta general**, y responde o te avisa cuando algo vence.

## Cómo funciona
1. Twilio recibe el mensaje de WhatsApp y lo reenvía a tu servidor (`/webhook`).
2. Claude clasifica el mensaje y extrae la fecha si aplica.
3. Se guarda en una base de datos SQLite (`reminders`, `notes`, `conversation_history`).
4. Un scheduler revisa cada minuto si hay recordatorios vencidos y te escribe por WhatsApp.

---

## 1. Consigue tus credenciales

### Anthropic (Claude)
1. Crea cuenta en https://console.anthropic.com/
2. Genera una API key.

### Twilio (sandbox de WhatsApp — gratis para pruebas)
1. Crea cuenta en https://www.twilio.com/try-twilio
2. Ve a **Messaging → Try it out → Send a WhatsApp message**.
3. Sigue las instrucciones para unirte al sandbox (envías un código desde tu WhatsApp al número que te dan).
4. Copia tu `Account SID` y `Auth Token` desde el dashboard principal.

---

## 2. Configura el proyecto localmente

```bash
git clone <este-repo>
cd memorae-clone
python -m venv .venv
source .venv/bin/activate      # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edita `.env` y pon tus claves reales.

## 3. Pruébalo localmente con ngrok

```bash
python app.py
```

En otra terminal:
```bash
ngrok http 5000
```

Copia la URL https que te da ngrok (ej. `https://abcd1234.ngrok.io`) y pégala en el
sandbox de Twilio como webhook: `https://abcd1234.ngrok.io/webhook`, método `POST`.

Ahora escríbele a tu bot desde WhatsApp (al número del sandbox de Twilio). Prueba:
- "Recuérdame llamar al dentista mañana a las 10am"
- "Anota que mi código de wifi es 1234"
- "¿Qué recordatorios tengo pendientes?"
- "¿Cuál es la capital de Australia?"

## 4. Desplegar en Render (gratis)

1. Sube este código a un repo de GitHub.
2. En https://render.com, crea un **New Web Service** y conecta el repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Agrega las variables de entorno del `.env` en la sección **Environment** de Render.
6. Una vez desplegado, copia la URL pública de Render (ej. `https://tu-app.onrender.com`)
   y actualiza el webhook en Twilio: `https://tu-app.onrender.com/webhook`.

⚠️ Nota: el plan gratuito de Render usa disco efímero — la base de datos SQLite se
puede borrar en cada redeploy. Para producción real, usa un disco persistente de
Render o cambia a Postgres (Render ofrece una base de datos Postgres gratuita).

## 5. Pasar del sandbox a un número real de WhatsApp

El sandbox de Twilio es solo para pruebas (los usuarios deben "unirse" con un código,
y expira). Para producción, solicita acceso a la API oficial de WhatsApp Business
dentro de Twilio (Twilio te guía en el proceso de verificación con Meta).

---

## Próximos pasos posibles
- Integración con Google Calendar (crear eventos reales, no solo recordatorios internos).
- Recordatorios recurrentes ("todos los lunes a las 9am").
- Multiidioma automático según el usuario.
- Exportar notas a PDF o a Notion.

¿Quieres que te ayude a agregar alguna de estas funciones? Solo dímelo.
