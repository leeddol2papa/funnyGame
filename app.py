import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import google.generativeai as genai
from collections import Counter

# 페이지 기본 설정
st.set_page_config(page_title="YOLO 객체 탐지 및 AI 해석", layout="wide")
st.title("YOLO 객체 탐지기 (with Gemini AI)")

# 1. Gemini API 키 고정 설정
# 🔒 주의: GitHub 리포지토리가 반드시 'Private(비공개)'일 때만 실제 키를 입력하세요!
GEMINI_API_KEY = "여기에_발급받은_실제_API_키를_입력하세요"

try:
    if GEMINI_API_KEY == "여기에_발급받은_실제_API_키를_입력하세요" or not GEMINI_API_KEY:
        st.sidebar.warning("⚠️ 코드가 수정되지 않았습니다. 변수에 실제 Gemini API 키를 입력해주세요.")
        gemini_ready = False
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        gemini_ready = True
except Exception as e:
    gemini_ready = False
    st.sidebar.error(f"⚠️ Gemini API 인증 실패: {e}")

# 2. 모델 로드 (캐싱하여 재로딩 방지)
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# 모델 경로 (깃헙 리포지토리 기준 상대 경로로 설정)
MODEL_PATH = "model/best.pt"

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"모델 파일을 찾을 수 없거나 로드에 실패했습니다: {e}")
    st.stop()

# 3. 이미지 업로드 UI
uploaded_file = st.file_uploader("탐지할 이미지를 업로드하세요", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # PIL을 사용하여 이미지 읽기
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("원본 이미지")
        st.image(image, use_container_width=True)

    # 파라미터 조정 UI
    conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.01, max_value=1.0, value=0.05, step=0.01)

    if st.sidebar.button("탐지 실행"):
        with st.spinner("이미지를 분석하고 있습니다..."):
            # 4. 모델 예측
            results = model.predict(source=image, conf=conf_threshold)
            r = results[0]
            
            # 5. 바운딩 박스 그려진 이미지 추출 (BGR -> RGB 변환)
            im_array = r.plot(line_width=2) 
            im_rgb = im_array[..., ::-1] 
            res_image = Image.fromarray(im_rgb)

            with col2:
                st.subheader("탐지 결과 이미지")
                st.image(res_image, use_container_width=True)

            # 6. 결과 정보 출력 (데이터프레임 형태)
            st.write("---")
            st.subheader("탐지된 객체 정보")
            st.write(f"**총 탐지된 객체 수:** {len(r.boxes)}")

            if len(r.boxes) > 0:
                det_data = []
                detected_classes = [] # Gemini 프롬프트용 클래스 리스트
                
                for i, box in enumerate(r.boxes):
                    class_id = int(box.cls.item())
                    class_name = model.names[class_id]
                    conf = float(box.conf.item())
                    xyxy = box.xyxy.cpu().numpy().ravel()
                    
                    detected_classes.append(class_name)
                    det_data.append({
                        "ID": i,
                        "클래스명": class_name,
                        "정확도(Conf)": f"{conf:.4f}",
                        "BBox (x1, y1, x2, y2)": f"[{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]"
                    })
                st.table(det_data)
                
                # 7. Gemini AI 결과 해석 기능
                if gemini_ready:
                    st.write("---")
                    st.subheader("🤖 Gemini AI 결과 해석")
                    
                    # 객체 개수 요약
                    class_counts = Counter(detected_classes)
                    summary_str = ", ".join([f"{k} {v}개" for k, v in class_counts.items()])
                    
                    prompt = f"""
                    당신은 이미지 분석 및 품질 관리/안전 환경 평가 전문가입니다.
                    다음은 YOLO 객체 탐지 모델이 이 이미지에서 탐지한 객체들의 목록입니다: {summary_str}
                    
                    제공된 원본 이미지 시각 정보와 텍스트 탐지 결과를 종합하여 다음을 수행해주세요:
                    1. 현장의 전반적인 상황 및 시각적 상태를 요약해주세요.
                    2. 탐지된 객체들의 관계나 작업 환경적 의미를 해석해주세요.
                    3. 현장 안전 기준이나 품질 관리 측면에서 주의 깊게 봐야 할 점이나 개선 인사이트를 도출해주세요.
                    """
                    
                    with st.spinner("Gemini가 시각 정보와 탐지 결과를 종합하여 해석하고 있습니다..."):
                        try:
                            # 멀티모달 분석 실행
                            response = gemini_model.generate_content([image, prompt])
                            st.markdown(response.text)
                        except Exception as e:
                            st.error(f"Gemini API 호출 중 오류가 발생했습니다: {e}")
