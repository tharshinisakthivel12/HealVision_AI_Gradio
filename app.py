import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "best_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("✅ Model Loaded Successfully")


# -----------------------------
# Prediction Function
# -----------------------------
def predict_wound(img):

    img = img.convert("RGB")
    img = img.resize((224,224))

    img = image.img_to_array(img)

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    prediction = float(model.predict(img, verbose=0)[0][0])

    if prediction >= 0.5:

        confidence = prediction*100

        return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━

🩺 AI DIAGNOSIS REPORT

━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Diagnosis

🟢 Healthy Skin

━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Confidence

{confidence:.2f} %

━━━━━━━━━━━━━━━━━━━━━━━━━━

🟢 Risk Level

LOW

━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 Recommendation

✔ Continue regular foot care.

✔ Keep feet clean and dry.

✔ Inspect feet daily.

✔ Wear comfortable footwear.

━━━━━━━━━━━━━━━━━━━━━━━━━━

Thank you for using HealVision AI ❤️
"""

    else:

        confidence=(1-prediction)*100

        return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━

🩺 AI DIAGNOSIS REPORT

━━━━━━━━━━━━━━━━━━━━━━━━━━

🚨 Diagnosis

🔴 Diabetic Foot Ulcer

━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Confidence

{confidence:.2f} %

━━━━━━━━━━━━━━━━━━━━━━━━━━

🔴 Risk Level

HIGH

━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 Recommendation

✔ Consult a healthcare professional immediately.

✔ Keep the wound clean.

✔ Avoid excessive pressure on the affected area.

✔ Monitor the wound daily.

━━━━━━━━━━━━━━━━━━━━━━━━━━

Thank you for using HealVision AI ❤️
"""


# -----------------------------
# Gradio Interface
# -----------------------------

demo = gr.Interface(

    fn=predict_wound,

    inputs=gr.Image(
        type="pil",
        label="📤 Upload Foot Wound Image",
        height=350
    ),

    outputs=gr.Textbox(
        label="🩺 AI Diagnosis Report",
        lines=18
    ),

    title="""
# 🩺 HealVision AI
## AI-Powered Diabetic Foot Ulcer Detection System
""",

    description="""
### Welcome to HealVision AI

HealVision AI is an AI-powered healthcare application that detects
**Diabetic Foot Ulcers** from wound images using the
**EfficientNetB0 Deep Learning Model**.

---

### How to Use

1️⃣ Upload a diabetic foot image.

2️⃣ Click **🚀 Analyze Image**

3️⃣ View the AI Diagnosis Report.

---

### Features

✅ Detects Healthy Skin

✅ Detects Diabetic Foot Ulcers

✅ Displays Confidence Score

✅ Provides Basic Medical Recommendations

---

⚠ **Disclaimer**

This AI application is developed for educational and research purposes only.

It should NOT replace diagnosis by a qualified medical professional.
""",

    submit_btn="🚀 Analyze Image",

    clear_btn="🗑 Clear",

    theme=gr.themes.Soft(),

    css="""
    footer{visibility:hidden;}
    .gradio-container{max-width:1200px !important;margin:auto;}
    h1{text-align:center;color:#2563EB;font-size:42px;font-weight:bold;}
    h2{text-align:center;color:#1E40AF;font-size:26px;}
    textarea{font-size:17px !important;}
    """

)

demo.launch(
    server_name="0.0.0.0",
    server_port=7860
)
