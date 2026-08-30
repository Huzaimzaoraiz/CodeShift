import re

with open('frontend/src/App.jsx', 'r') as f:
    content = f.read()

replacements = {
    "✅ Feedback persisted!": "Feedback persisted!",
    "❌ Error submitting": "Error submitting",
    "⚙️ ": "",
    "👤 ": "",
    "✓ JWT Token Secured": "JWT Token Secured",
    "🤖 ": "",
    "🏢 ": "",
    "➕": "+",
    "📈 ": "",
    "🔍 ": "",
    "📄 ": "",
    "⬇": "v",
    "✅ Data fetched": "Data fetched",
    "🎯 ": "",
    "📐 ": "",
    "📊 ": "",
    "🗳️ ": "",
    "🌲 ": "",
    "📉 ": "",
    "🔴 Anomaly": "[Anomaly]",
    "🟢 Normal": "[Normal]",
    "🔬 ": "",
    "✅ Statistically Significant": "[Statistically Significant]",
    "⚠️ Not Significant": "[Not Significant]",
    "🧬 ": "",
    "🧠 ": "",
    "📖 ": "",
    "🔗 ": "",
    "⚠️ ": "",
    "🚨 ": "",
    "🔄 ": ""
}

for k, v in replacements.items():
    content = content.replace(k, v)

with open('frontend/src/App.jsx', 'w') as f:
    f.write(content)

print("Emojis stripped.")
