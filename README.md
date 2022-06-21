# PROJECT COOKIE!

## 요구사항 
    python3.8버전
    
    요구사항 > requirements.txt

    tensorflow-gpu 설정
    conda install cudatoolkit
    conda install cudnn
    pip install tensorflow-gpu

    media 폴더에 ani 폴더 생성해주기
    최상위 폴더에 model 폴더 생성하고 모델 2개 넣어주기

    python manage.py migrate

<br>

# 기능

흑백 만화책 이미지에서 자동으로 컷과 말풍선을 분리하고, 애니메이션으로 만들어줍니다.

만들어진 애니메이션은 저장 가능합니다.

## 설정 가능한 기능

    1. 컷의 등장순서
         읽기방식 좌 > 우 혹은 우 > 좌로 설정 가능합니다.

    2. 컷의 출현 방식
        컷 방식은 웹툰처럼 단일 컷이 순서대로 출현합니다.
        만화책 방식은 하나의 비어있는 페이지 바탕에서 컷이 순서대로 출현합니다.

    3. 애니메이션
        현재 말풍선 강조 효과만 가능합니다.
        각 말풍선이 순서대로 줌인-줌아웃 됩니다.
        말풍선 내부의 글자 수에 따라서 강조되어있는 시간이 달라집니다.
    
    4. 영상 전환 효과
        컷 방식에서만 작동합니다.
        개별 컷이 어느 방향으로 넘어갈지, 디졸브 방식으로 사라질지를 설정 가능합니다.

<br>

# 실행화면

<image src='![Cookie](https://user-images.githubusercontent.com/79143006/174873054-4604f335-778b-4035-858d-1f15a86d6119.gif)
'>