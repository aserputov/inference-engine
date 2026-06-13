import torch
import tiktoken
import time
import json
from flask import Flask, request, jsonify, render_template_string, Response, stream_with_context
from engine import GPT2Cached, load_openai_weights, generate_streaming

model = GPT2Cached()
load_openai_weights(model)
model.eval()
enc = tiktoken.get_encoding("gpt2")

print("Ready!\n")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Inference Engine</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'SF Mono', 'Fira Code', monospace;
            background: #0a0a0a; color: #e0e0e0;
            display: flex; justify-content: center; padding-top: 60px; min-height: 100vh;
        }
        .container { width: 720px; }
        h1 { font-size: 22px; margin-bottom: 4px; letter-spacing: -0.5px; }
        .subtitle { color: #666; margin-bottom: 32px; font-size: 13px; }
        .label {
            font-size: 10px; color: #555; text-transform: uppercase;
            letter-spacing: 1.5px; margin-bottom: 6px;
        }
        textarea {
            width: 100%; padding: 14px 18px; font-size: 15px;
            border: 1px solid #222; border-radius: 8px;
            background: #111; color: #fff; outline: none;
            font-family: inherit; resize: vertical; min-height: 80px;
        }
        textarea:focus { border-color: #333; }
        .controls {
            display: flex; gap: 12px; margin-top: 12px; align-items: center;
            flex-wrap: wrap;
        }
        .control-group { display: flex; flex-direction: column; gap: 4px; }
        .control-group label { font-size: 10px; color: #555; text-transform: uppercase; letter-spacing: 1px; }
        .control-group input {
            padding: 8px 12px; font-size: 13px; border: 1px solid #222;
            border-radius: 6px; background: #111; color: #fff;
            font-family: inherit; outline: none; width: 100px;
        }
        button {
            padding: 10px 24px; font-size: 14px; border: 1px solid #333;
            border-radius: 8px; background: #1a1a1a; color: #fff;
            cursor: pointer; font-family: inherit; transition: all 0.15s;
            margin-left: auto;
        }
        button:hover { background: #252525; border-color: #444; }
        button:disabled { opacity: 0.4; cursor: not-allowed; }
        .output-section { margin-top: 24px; }
        .output {
            padding: 14px 18px; background: #111; border-radius: 8px;
            border: 1px solid #1a1a1a; min-height: 120px; font-size: 15px;
            line-height: 1.8; white-space: pre-wrap;
        }
        .prompt-text { color: #888; }
        .generated-text { color: #48bfe3; }
        .cursor { animation: blink 1s infinite; color: #48bfe3; }
        @keyframes blink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0; } }
        .stats {
            margin-top: 12px; padding: 10px 14px; background: #0d0d0d;
            border-radius: 6px; border: 1px solid #151515;
            font-size: 11px; color: #444; display: none;
        }
        .stats span { color: #48bfe3; }
        .examples { margin-top: 32px; }
        .examples p {
            color: #444; font-size: 12px; cursor: pointer; padding: 3px 0;
            transition: color 0.15s;
        }
        .examples p:hover { color: #999; }
        .arch {
            margin-top: 32px; padding: 14px; background: #0d0d0d;
            border-radius: 8px; font-size: 11px; color: #333; line-height: 1.8;
            border: 1px solid #151515;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Inference Engine</h1>
        <p class="subtitle">GPT-2 124M &middot; KV-Cache &middot; Streaming &middot; Token-by-token generation</p>

        <div class="label">Prompt</div>
        <textarea id="input" placeholder="Type a prompt...">The meaning of life is</textarea>

        <div class="controls">
            <div class="control-group">
                <label>Tokens</label>
                <input type="number" id="max_tokens" value="80" min="1" max="300">
            </div>
            <div class="control-group">
                <label>Temperature</label>
                <input type="number" id="temperature" value="0.8" min="0.1" max="2.0" step="0.1">
            </div>
            <div class="control-group">
                <label>Top-K</label>
                <input type="number" id="top_k" value="40" min="0" max="100">
            </div>
            <button id="btn" onclick="doGenerate()">Generate</button>
        </div>

        <div class="output-section">
            <div class="label">Output <span style="color:#333">&middot; streaming with KV-cache</span></div>
            <div class="output" id="output"><span style="color:#333">...</span></div>
        </div>

        <div class="stats" id="stats"></div>

        <div class="examples">
            <div class="label">Try these</div>
            <p onclick="tryExample(this)">In a shocking finding, scientists discovered</p>
            <p onclick="tryExample(this)">Once upon a time, in a land far away,</p>
            <p onclick="tryExample(this)">The best programming language is</p>
            <p onclick="tryExample(this)">Artificial intelligence will</p>
            <p onclick="tryExample(this)">Dear diary, today I learned that</p>
        </div>

        <div class="arch">
            GPT-2 Small (124M) with KV-Cache optimization
            &middot; Prefill: process entire prompt in one pass, cache K/V
            &middot; Decode: generate one token at a time, reuse cached K/V
            &middot; Streaming: tokens sent via SSE as they're generated
        </div>
    </div>
    <script>
        const input = document.getElementById('input');
        const output = document.getElementById('output');
        const btn = document.getElementById('btn');
        const stats = document.getElementById('stats');

        async function doGenerate() {
            const prompt = input.value.trim();
            if (!prompt) return;

            btn.disabled = true;
            btn.textContent = 'Generating...';
            stats.style.display = 'none';
            output.innerHTML = '<span class="prompt-text">' + escapeHtml(prompt) + '</span><span class="cursor">|</span>';

            const startTime = performance.now();
            let tokenCount = 0;
            let firstTokenTime = null;
            let generatedText = '';

            try {
                const res = await fetch('/stream', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        prompt,
                        max_tokens: parseInt(document.getElementById('max_tokens').value),
                        temperature: parseFloat(document.getElementById('temperature').value),
                        top_k: parseInt(document.getElementById('top_k').value),
                    })
                });

                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {
                    const {done, value} = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, {stream: true});
                    const lines = buffer.split('\\n');
                    buffer = lines.pop();

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = JSON.parse(line.slice(6));
                            if (data.token) {
                                if (firstTokenTime === null) firstTokenTime = performance.now();
                                tokenCount++;
                                generatedText += data.token;
                                output.innerHTML =
                                    '<span class="prompt-text">' + escapeHtml(prompt) + '</span>' +
                                    '<span class="generated-text">' + escapeHtml(generatedText) + '</span>' +
                                    '<span class="cursor">|</span>';
                            }
                        }
                    }
                }

                output.innerHTML =
                    '<span class="prompt-text">' + escapeHtml(prompt) + '</span>' +
                    '<span class="generated-text">' + escapeHtml(generatedText) + '</span>';

                const totalTime = (performance.now() - startTime) / 1000;
                const ttft = firstTokenTime ? ((firstTokenTime - startTime) / 1000) : 0;
                const decodeTime = totalTime - ttft;
                const tps = tokenCount > 0 ? (tokenCount / decodeTime) : 0;

                stats.style.display = 'block';
                stats.innerHTML =
                    'Prefill (TTFT): <span>' + ttft.toFixed(2) + 's</span> &middot; ' +
                    'Decode: <span>' + decodeTime.toFixed(2) + 's</span> &middot; ' +
                    'Tokens: <span>' + tokenCount + '</span> &middot; ' +
                    'Speed: <span>' + tps.toFixed(1) + ' tokens/sec</span> &middot; ' +
                    'Total: <span>' + totalTime.toFixed(2) + 's</span>';
            } catch(e) {
                output.innerHTML = '<span style="color:#663333">Error: ' + e.message + '</span>';
            }

            btn.disabled = false;
            btn.textContent = 'Generate';
        }

        function tryExample(el) {
            input.value = el.textContent;
            doGenerate();
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                doGenerate();
            }
        });
    </script>
</body>
</html>
"""

app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/stream", methods=["POST"])
def stream_api():
    data = request.json
    prompt = data.get("prompt", "").strip()
    max_tokens = min(data.get("max_tokens", 80), 300)
    temperature = max(0.1, min(data.get("temperature", 0.8), 2.0))
    top_k = max(0, min(data.get("top_k", 40), 100))

    if not prompt:
        return jsonify({"error": "Empty prompt"})

    tokens = enc.encode(prompt)

    def stream():
        for token_id in generate_streaming(model, tokens, max_tokens=max_tokens,
                                           temperature=temperature, top_k=top_k):
            word = enc.decode([token_id])
            yield f"data: {json.dumps({'token': word})}\n\n"

    return Response(stream_with_context(stream()), mimetype='text/event-stream')

if __name__ == "__main__":
    print("Starting server at http://localhost:5004")
    app.run(port=5004, debug=False)
