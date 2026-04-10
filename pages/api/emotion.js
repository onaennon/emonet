export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const { message, systemPrompt } = req.body;
  const apiKey = process.env.GEMINI_API_KEY;

  if (!apiKey) {
    return res.status(500).json({ error: "NO_API_KEY" });
  }

  const contents = [
    { role: "user", parts: [{ text: systemPrompt }] },
    { role: "model", parts: [{ text: "Understood." }] },
    { role: "user", parts: [{ text: message }] }
  ];

  try {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${apiKey}`;
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents,
        generationConfig: { maxOutputTokens: 500, temperature: 0.1 }
      })
    });

    const data = await response.json();

    if (data.error) {
      return res.status(500).json({ error: data.error.message, code: data.error.code });
    }

    const text = data.candidates?.[0]?.content?.parts?.[0]?.text || "";
    return res.status(200).json({ text });
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
}
