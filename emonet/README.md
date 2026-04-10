# EmoNet — Neural Emotion State Tracker

An AI chat app with integrated emotion tracking. Chat with Gemini while a neural network analyzes your emotional state in real-time, visualized as a swirling color dot.

## Setup

### 1. Get a Gemini API Key
- Go to [ai.google.dev](https://ai.google.dev)
- Sign in with your Google account
- Click "Get API Key" in Google AI Studio
- Copy the key

### 2. Local Development
```bash
npm install
```

Create a `.env.local` file in the project root:
```
GEMINI_API_KEY=your_api_key_here
```

Run the dev server:
```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) on your phone or browser.

### 3. Deploy to Vercel

1. Push this repo to GitHub
2. Go to [vercel.com](https://vercel.com) and sign in with GitHub
3. Click "Add New Project" → import your `emonet` repo
4. Go to **Settings → Environment Variables**
5. Add `GEMINI_API_KEY` with your key as the value
6. Click **Deploy**

## Project Structure

```
emonet/
├── pages/
│   ├── index.jsx          # Main app (chat + emotion tracker)
│   ├── _app.js            # App wrapper with global styles
│   └── api/
│       ├── chat.js        # Proxies chat messages to Gemini
│       └── emotion.js     # Proxies emotion analysis to Gemini
├── styles/
│   └── globals.css        # Global styles
├── package.json
├── next.config.js
├── .env.local.example     # Template for API key
└── .gitignore
```

## How It Works

- Each message you send triggers two API calls:
  1. **Chat call** — normal conversation with Gemini
  2. **Emotion call** — silent analysis of your message across 42 emotion dimensions
- The emotion network tracks your state over time and learns to predict where your mood is heading
- The swirling dot visualizes your predicted emotional state — colors correspond to emotions
- All data stays on your device (localStorage)
- Optional: enable "Emotion-Aware Responses" to let Gemini adjust its tone based on your detected state
