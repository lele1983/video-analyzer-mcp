# 🎬 Video Analyzer MCP Server

Un server MCP (Model Context Protocol) per scaricare e analizzare video da Instagram e TikTok usando le API di Apify e Google Gemini Flash.

## ✨ Funzionalità

- **Download Video Instagram**: Scarica Reel, Post e Story da Instagram
- **Download Video TikTok**: Scarica video TikTok (senza watermark)
- **Analisi AI con Gemini**: Analisi completa dei contenuti video
- **Workflow Combinato**: Download + Analisi in un singolo comando

## 🛠️ Tools Disponibili

| Tool | Descrizione |
|------|-------------|
| `video_download_instagram` | Scarica video da Instagram (Reel/Post/Story) |
| `video_download_tiktok` | Scarica video da TikTok |
| `video_analyze` | Analizza un video con Gemini Flash AI |
| `video_download_and_analyze` | Workflow completo: download + analisi |

### Tipi di Analisi

| Tipo | Descrizione |
|------|-------------|
| `full` | Analisi completa (contenuto, sentiment, target, viralità) |
| `summary` | Riassunto rapido in 3-5 frasi |
| `transcript` | Trascrizione del parlato e testo visibile |
| `content` | Descrizione dettagliata del contenuto visivo |
| `sentiment` | Analisi del tono e impatto emotivo |

## 📦 Installazione

### 1. Clona/Copia i file

```bash
# Crea la directory
mkdir -p ~/mcp-servers/video-analyzer
cd ~/mcp-servers/video-analyzer

# Copia i file del progetto
# server.py, requirements.txt, .env.example
```

### 2. Installa le dipendenze

```bash
# Con pip
pip install -r requirements.txt

# Oppure con uv (consigliato)
uv pip install -r requirements.txt
```

### 3. Configura le API Keys

```bash
# Copia il file di esempio
cp .env.example .env

# Modifica con le tue chiavi
nano .env
```

**Ottieni le chiavi:**
- **Apify**: [console.apify.com/account/integrations](https://console.apify.com/account/integrations)
- **Gemini**: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

## ⚙️ Configurazione MCP Client

### Claude Desktop (claude_desktop_config.json)

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "video-analyzer": {
      "command": "python",
      "args": ["/path/to/video-analyzer-mcp/server.py"],
      "env": {
        "APIFY_API_TOKEN": "your_apify_token",
        "GEMINI_API_KEY": "your_gemini_key"
      }
    }
  }
}
```

### Con uv (consigliato per isolamento)

```json
{
  "mcpServers": {
    "video-analyzer": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/path/to/video-analyzer-mcp",
        "python", "server.py"
      ],
      "env": {
        "APIFY_API_TOKEN": "your_apify_token",
        "GEMINI_API_KEY": "your_gemini_key"
      }
    }
  }
}
```

### Cline / VS Code MCP Extension

Aggiungi al tuo `settings.json` o configurazione MCP:

```json
{
  "mcp.servers": {
    "video-analyzer": {
      "command": "python",
      "args": ["/path/to/server.py"],
      "env": {
        "APIFY_API_TOKEN": "...",
        "GEMINI_API_KEY": "..."
      }
    }
  }
}
```

## 🚀 Utilizzo

### Esempio 1: Download + Analisi Completa

```
Analizza questo video TikTok: https://www.tiktok.com/@user/video/123456
```

Il modello userà `video_download_and_analyze` per:
1. Rilevare automaticamente la piattaforma
2. Scaricare il video via Apify
3. Analizzarlo con Gemini Flash
4. Restituire metadati + analisi AI

### Esempio 2: Solo Download

```
Scarica questo reel Instagram: https://www.instagram.com/reel/ABC123/
```

Restituisce JSON con:
- URL diretto del video
- Thumbnail
- Metriche (like, commenti, views)
- Caption e autore

### Esempio 3: Analisi con Prompt Custom

```
Analizza questo video cercando:
- Prodotti mostrati e brand visibili
- Call-to-action presenti
- Stima del budget di produzione

URL: https://www.tiktok.com/@brand/video/789
```

### Esempio 4: Solo Trascrizione

```
Trascrivi tutto il parlato di questo video Instagram
URL: https://www.instagram.com/p/XYZ/
Tipo analisi: transcript
```

### Esempio 5: Analisi Batch (Parallela)

```
Analizza questi video in parallelo:
1. https://www.tiktok.com/@user/video/111
2. https://www.instagram.com/reel/222
3. https://www.tiktok.com/@user/video/333
```

Il modello userà `batch_video_analyze` per scaricare e analizzare tutti i video contemporaneamente, restituendo un report unico.

## 📊 Output Examples

### Download Response (JSON)

```json
{
  "status": "success",
  "platform": "tiktok",
  "video_url": "https://...",
  "thumbnail_url": "https://...",
  "caption": "Video caption here...",
  "author": "username",
  "likes": 125000,
  "comments": 3200,
  "shares": 8500,
  "plays": 2500000,
  "duration": 45,
  "music": {
    "title": "Original Sound",
    "author": "Creator Name"
  }
}
```

### Analysis Response (Markdown)

```markdown
# Video Analysis: TikTok

## Metadata
- **Autore**: @username
- **Piattaforma**: tiktok
- **Like**: 125,000
- **Views**: 2,500,000

## AI Analysis

### Riassunto
Video che mostra un tutorial di cucina rapida...

### Contenuto Visivo
Ambientazione in cucina moderna...

### Sentiment
Tono: Informativo e friendly
Score: 8/10 (positivo)
...
```

## 🔧 Troubleshooting

### Errore: "APIFY_API_TOKEN environment variable is required"
→ Assicurati di aver configurato le variabili d'ambiente nel client MCP

### Errore: "No video found at this URL"
→ Il video potrebbe essere privato, eliminato, o l'URL non è valido

### Errore: "Video file is too large"
→ Gemini ha un limite di ~50MB per video. Prova con video più corti

### Errore: "Rate limit exceeded"
→ Aspetta qualche minuto prima di riprovare (limiti API)

## 📋 Costi API

| Servizio | Costo |
|----------|-------|
| **Apify** | ~$0.25-0.50 per 1000 video scaricati |
| **Gemini Flash** | ~$0.075 per 1M token input |

*Prezzi indicativi, verifica sempre i pricing ufficiali*

## 🛡️ Note sulla Privacy

- I video vengono elaborati in memoria e non salvati permanentemente
- Le API di terze parti (Apify, Google) potrebbero loggare le richieste
- Non utilizzare per scaricare contenuti protetti da copyright senza autorizzazione

## 📄 Licenza

MIT License - Usa liberamente, ma rispetta i ToS delle piattaforme social.

---

**Creato per l'integrazione con Claude e altri LLM via MCP** 🤖
