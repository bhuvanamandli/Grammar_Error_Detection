import base64

import torch
import streamlit as st
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForTokenClassification


def load_grammar_model():
    # Token-classification model + tokenizer, saved earlier with:
    #   model.save_pretrained("distilbert_ged_token_model1")
    #   tokenizer.save_pretrained("distilbert_ged_token_model1")
    tokenizer = AutoTokenizer.from_pretrained("distilbert_ged_token_model1")
    model = AutoModelForTokenClassification.from_pretrained("distilbert_ged_token_model1")
    model.eval()

    # id2label ("CORRECT" / "INCORRECT") is already saved inside the model's
    # config.json, so no separate label encoder file is needed here.
    return model, tokenizer


model, tokenizer = load_grammar_model()


def find_grammar_errors(text):
    words = text.split()
    tokens = tokenizer(words, return_tensors="pt", is_split_into_words=True, truncation=True)

    with torch.no_grad():
        logits = model(**tokens).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    predicted_ids = probabilities.argmax(dim=-1).tolist()
    word_ids = tokens.word_ids(batch_index=0)

    # keep only the first subword's prediction (and confidence) for each word
    word_labels = {}
    word_confidences = {}
    for pos, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx not in word_labels:
            predicted_id = predicted_ids[pos]
            word_labels[word_idx] = model.config.id2label[predicted_id]
            word_confidences[word_idx] = probabilities[pos][predicted_id].item()

    flagged_indices = [i for i, label in word_labels.items() if label == "INCORRECT"]
    overall = "INCORRECT" if flagged_indices else "CORRECT"
    # overall confidence = the lowest confidence among flagged words (or, if
    # correct, the average confidence across all words)
    if flagged_indices:
        confidence = min(word_confidences[i] for i in flagged_indices)
    else:
        confidence = sum(word_confidences.values()) / len(word_confidences)

    return {
        "overall": overall,
        "confidence": confidence,
        "words": words,
        "word_labels": word_labels,
        "word_confidences": word_confidences,
        "flagged_indices": flagged_indices,
    }


def render_highlighted_sentence(result):
    """Build an HTML string with incorrect words highlighted in red."""
    spans = []
    for i, word in enumerate(result["words"]):
        if i in result["flagged_indices"]:
            spans.append(
                f'<span style="background-color:#ffd6d6; color:#8a1f1f; '
                f'padding:2px 6px; border-radius:4px; font-weight:600;">{word}</span>'
            )
        else:
            spans.append(word)
    return " ".join(spans)


def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


img = get_base64("grammar.jpg")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{img}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Grammar Error Detection ✍️")

st.write(
    "Enter a sentence to check its grammar and find the incorrect word."
)

text = st.text_area(
    "Enter your text:",
    placeholder="Example: She go to school every day"
)

try:
    if st.button("Check Grammar"):

        if text.strip() == "":
            st.warning("Please enter some text.")

        else:
            result = find_grammar_errors(text)
            
            print('RESULT~~', result)

            col1, col2 = st.columns(2)

            with col1:
                if result["overall"] == "CORRECT":
                    st.success(f"Prediction: **{result['overall']}**")
                else:
                    st.error(f"Prediction: **{result['overall']}**")
            with col2:
                st.info(f"🎯 Confidence: **{result['confidence'] * 100:.2f}%**")

            col1, col2 = st.columns(2)
            with col1:
                if result["flagged_indices"]:
                    flagged_words = [result["words"][i] for i in result["flagged_indices"]]
                    st.write(f"Likely problem word(s): **{', '.join(flagged_words)}**")

                st.markdown(render_highlighted_sentence(result), unsafe_allow_html=True)

                with st.expander("See per-word breakdown"):
                    for i, word in enumerate(result["words"]):
                        label = result["word_labels"].get(i, "?")
                        conf = result["word_confidences"].get(i, 0.0)
                        st.write(f"`{word}` → {label} ({conf * 100:.1f}%)")

except Exception as e:
    st.error(f"An error occurred: {e}")