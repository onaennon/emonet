export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });
  const { messages, systemPrompt } = req.body;
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) return res.status(500).json({ error: "NO_API_KEY" });

  const contents = [];
  if (systemPrompt) {
    contents.push({ role: "user", parts: [{ text: systemPrompt }] });
    contents.push({ role: "model", parts: [{ text: "Understood." }] });
  }
  for (const m of messages) {
    contents.push({ role: m.role === "assistant" ? "model" : "user", parts: [{ text: m.text }] });
  }

  try {
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key=${apiKey}`,
      { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ contents, generationConfig: { maxOutputTokens: 1024, temperature: 0.7 } }) }
    );
    const data = await response.json();
    if (data.error) return res.status(500).json({ error: data.error.message, code: data.error.code });
    const text = data.candidates?.[0]?.content?.parts?.[0]?.text || "";
    return res.status(200).json({ text });
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
}
