from __future__ import annotations

from pathlib import Path

import streamlit as st

from inference_utils import RobustInferenceEngine, project_root_from_demo_file


st.set_page_config(
    page_title="ViEduFeedback Classifier",
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource(show_spinner=True)
def load_engine(root_path: str) -> RobustInferenceEngine:
    return RobustInferenceEngine(Path(root_path))


def show_scores(title: str, scores: dict) -> None:
    if not scores:
        st.caption(f"{title}: mô hình này không có probability thật; confidence là pseudo-score từ decision margin.")
        return

    st.markdown(f"**{title}**")
    for label, score in scores.items():
        st.progress(float(score), text=f"{label}: {score:.3f}")


def main() -> None:
    root = project_root_from_demo_file(__file__)

    st.title("🎓 ViEduFeedback Robust Classifier")
    st.caption("Dự đoán cảm xúc và chủ đề phản hồi sinh viên tiếng Việt")

    with st.spinner("Đang load model..."):
        engine = load_engine(str(root))

    with st.sidebar:
        st.header("Thiết lập")

        mode = st.radio(
            "Chế độ suy luận",
            options=["auto_router", "phobert_only", "char_svm_only"],
            format_func=lambda value: {
                "auto_router": "Auto Router: PhoBERT + char SVM fallback",
                "phobert_only": "PhoBERT only",
                "char_svm_only": "TF-IDF char SVM only",
            }[value],
            index=0,
        )

        st.markdown("---")
        st.write("Detector threshold:", engine.threshold)
        st.caption("Threshold lấy từ Stage 8 validation.")

        with st.expander("Trạng thái model"):
            if engine.load_errors:
                for name, error in engine.load_errors.items():
                    if ("phobert" in name and name.split("_")[-1] in engine.phobert) or (
                        "char_svm" in name and name.split("_")[-1] in engine.char_svm
                    ):
                        continue
                    st.warning(f"{name}: {error}")
            st.write("PhoBERT loaded:", sorted(engine.phobert.keys()))
            st.write("Char SVM loaded:", sorted(engine.char_svm.keys()))

    examples = {
        "Phản hồi chuẩn, tích cực": "giảng viên rất nhiệt tình, bài giảng dễ hiểu và có nhiều ví dụ thực tế",
        "Phản hồi chuẩn, tiêu cực": "phòng học nóng, máy chiếu mờ và âm thanh rất khó nghe",
        "Thiếu dấu": "giang vien rat nhiet tinh nhung slide hoi kho hieu",
        "Teencode": "gv day ok nhung phong hoc hoi nong",
    }

    selected_example = st.selectbox("Chọn ví dụ nhanh", list(examples.keys()))
    default_text = examples[selected_example]

    text = st.text_area(
        "Nhập phản hồi của sinh viên",
        value=default_text,
        height=130,
        placeholder="Ví dụ: giảng viên dạy dễ hiểu nhưng phòng học hơi nóng",
    )

    predict_clicked = st.button("Dự đoán", type="primary")

    if not predict_clicked:
        st.info("Nhập phản hồi rồi bấm **Dự đoán**.")
        return

    text = text.strip()
    if not text:
        st.error("Vui lòng nhập nội dung phản hồi.")
        return

    try:
        result = engine.predict_both(text, mode=mode)
    except Exception as exc:
        st.error("Không thể dự đoán. Kiểm tra lại model artifacts.")
        st.exception(exc)
        return

    detector = result["detector"]
    sentiment = result["sentiment"]
    topic = result["topic"]

    st.subheader("1. Phân tích đầu vào")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số ký tự chữ", detector["alpha_chars"])
    c2.metric("Số từ", detector["word_count"])
    c3.metric("Ký tự tiếng Việt có dấu", detector["marked_vietnamese_chars"])
    c4.metric("Accented ratio", f"{detector['accented_ratio']:.4f}")

    if detector["suspected_no_accent"]:
        st.warning("Detector: phản hồi có khả năng **thiếu dấu** → ưu tiên TF-IDF char SVM trong Auto Router.")
    else:
        st.success("Detector: phản hồi không bị xem là thiếu dấu → ưu tiên PhoBERT trong Auto Router.")

    st.subheader("2. Kết quả dự đoán")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Cảm xúc")
        st.metric("Nhãn", sentiment.label_vi)
        st.write("Model dùng:", sentiment.model_used)
        if sentiment.confidence is not None:
            st.write("Confidence:", f"{sentiment.confidence:.4f}")
        show_scores("Điểm từng lớp", sentiment.scores)

    with col2:
        st.markdown("### Chủ đề")
        st.metric("Nhãn", topic.label_vi)
        st.write("Model dùng:", topic.model_used)
        if topic.confidence is not None:
            st.write("Confidence:", f"{topic.confidence:.4f}")
        show_scores("Điểm từng lớp", topic.scores)

    st.subheader("3. JSON kết quả")

    st.json(
        {
            "text": result["text"],
            "mode": result["mode"],
            "detector": result["detector"],
            "sentiment": {
                "label_id": sentiment.label_id,
                "label_en": sentiment.label_en,
                "label_vi": sentiment.label_vi,
                "model_used": sentiment.model_used,
                "confidence": sentiment.confidence,
            },
            "topic": {
                "label_id": topic.label_id,
                "label_en": topic.label_en,
                "label_vi": topic.label_vi,
                "model_used": topic.model_used,
                "confidence": topic.confidence,
            },
        }
    )


if __name__ == "__main__":
    main()
