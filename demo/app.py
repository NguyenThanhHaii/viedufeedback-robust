from __future__ import annotations

from pathlib import Path

import streamlit as st

from inference_utils import RobustInferenceEngine, project_root_from_demo_file


st.set_page_config(
    page_title="Phân loại phản hồi sinh viên",
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource(show_spinner=True)
def load_engine(root_path: str) -> RobustInferenceEngine:
    return RobustInferenceEngine(Path(root_path))


def show_scores(title: str, scores: dict) -> None:
    if not scores:
        st.caption("LinearSVC không trả xác suất thật; điểm hiển thị là score quy đổi từ margin.")
        return

    st.markdown(f"**{title}**")
    for label, score in scores.items():
        st.progress(float(score), text=f"{label}: {score:.3f}")


def main() -> None:
    root = project_root_from_demo_file(__file__)

    st.title("Phân loại phản hồi sinh viên")
    st.caption("Dự đoán cảm xúc và chủ đề từ phản hồi tiếng Việt.")

    with st.spinner("Đang tải mô hình..."):
        engine = load_engine(str(root))

    with st.sidebar:
        st.header("Tùy chọn")

        mode = st.radio(
            "Chế độ",
            options=["auto_router", "phobert_only", "char_svm_only"],
            format_func=lambda value: {
                "auto_router": "Tự động chọn mô hình",
                "phobert_only": "Chỉ dùng PhoBERT",
                "char_svm_only": "Chỉ dùng char SVM",
            }[value],
            index=0,
        )

        st.markdown("---")
        st.write("Ngưỡng phát hiện thiếu dấu:", engine.threshold)

        with st.expander("Model đã tải"):
            st.write("PhoBERT:", sorted(engine.phobert.keys()))
            st.write("Char SVM:", sorted(engine.char_svm.keys()))

            visible_errors = {
                name: error
                for name, error in engine.load_errors.items()
                if not (
                    name.startswith("phobert_") and name.replace("phobert_", "") in engine.phobert
                )
                and not (
                    name.startswith("char_svm_")
                    and name.replace("char_svm_", "") in engine.char_svm
                )
            }

            if visible_errors:
                for name, error in visible_errors.items():
                    st.warning(f"{name}: {error}")

    examples = {
        "Tích cực": "giảng viên rất nhiệt tình, bài giảng dễ hiểu và có nhiều ví dụ thực tế",
        "Tiêu cực": "phòng học nóng, máy chiếu mờ và âm thanh rất khó nghe",
        "Thiếu dấu": "giang vien rat nhiet tinh nhung slide hoi kho hieu",
        "Viết tắt": "gv day ok nhung phong hoc hoi nong",
    }

    selected_example = st.selectbox("Ví dụ nhanh", list(examples.keys()))
    text = st.text_area(
        "Nhập phản hồi",
        value=examples[selected_example],
        height=130,
        placeholder="Ví dụ: giảng viên dạy dễ hiểu nhưng phòng học hơi nóng",
    )

    if not st.button("Dự đoán", type="primary"):
        st.stop()

    text = text.strip()
    if not text:
        st.error("Vui lòng nhập nội dung phản hồi.")
        st.stop()

    try:
        result = engine.predict_both(text, mode=mode)
    except Exception as exc:
        st.error("Không thể dự đoán. Hãy kiểm tra lại model trong `outputs/models/`.")
        st.exception(exc)
        st.stop()

    detector = result["detector"]
    sentiment = result["sentiment"]
    topic = result["topic"]

    st.subheader("Thông tin đầu vào")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Số ký tự chữ", detector["alpha_chars"])
    col2.metric("Số từ", detector["word_count"])
    col3.metric("Ký tự có dấu", detector["marked_vietnamese_chars"])
    col4.metric("Tỷ lệ ký tự có dấu", f"{detector['accented_ratio']:.4f}")

    if detector["suspected_no_accent"]:
        st.warning("Câu này có khả năng thiếu dấu. Ở chế độ tự động, hệ thống dùng char SVM.")
    else:
        st.success("Câu này không bị xem là thiếu dấu. Ở chế độ tự động, hệ thống dùng PhoBERT.")

    st.subheader("Kết quả")

    left, right = st.columns(2)

    with left:
        st.markdown("### Cảm xúc")
        st.metric("Nhãn dự đoán", sentiment.label_vi)
        st.write("Model:", sentiment.model_used)
        if sentiment.confidence is not None:
            st.write("Điểm tin cậy:", f"{sentiment.confidence:.4f}")
        show_scores("Điểm từng lớp", sentiment.scores)

    with right:
        st.markdown("### Chủ đề")
        st.metric("Nhãn dự đoán", topic.label_vi)
        st.write("Model:", topic.model_used)
        if topic.confidence is not None:
            st.write("Điểm tin cậy:", f"{topic.confidence:.4f}")
        show_scores("Điểm từng lớp", topic.scores)

    with st.expander("JSON kết quả"):
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
