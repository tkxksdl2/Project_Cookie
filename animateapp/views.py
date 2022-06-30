from django.http import HttpResponseRedirect
from django.urls import reverse
from django.views.generic import CreateView, DetailView

from animateapp.forms import AnimateForm
from animateapp.models import Animate, AnimateImage

from PIL import Image

import numpy as np
import base64
import gc
import tensorflow as tf

from animateapp.funcToon.cuts import make_cut_bubble, image_preproc
from animateapp.funcToon.makeFrame import ComicFrameBook
from animateapp.funcComic.cuts import make_page_cut
from animateapp.funcVideo.video import view_seconds

# 충돌 오류 때문에 기록...
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

IMAGE_SIZE = 224


class AnimateCreateView(CreateView):
    model = Animate
    form_class = AnimateForm
    template_name = 'animateapp/create.html'

    def post(self, request, *args, **kwargs):
        # form 받아옴
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        # 이미지 파일들만 받아오기
        files = request.FILES.getlist('image')
        # Animate 모델에는 첫이미지만 저장되게 하기
        form.instance.image = files[0]
        animate = form.save(commit=False)
        animate.save()

        if form.is_valid():
            # 이미지 파일 np array 형식으로 저장하기위한 리스트 생성
            image_list = []
            # 모델 돌리고 정리된 이미지들 들어오는 부분
            img_list = []
            # 선택값들 받아오기
            l_r = str(form.instance.left_right)
            t_c = str(form.instance.toon_comic)
            ani_effect = str(form.instance.ani_effect)
            tran_effect = str(form.instance.transition_effect)

            # 이미지 파일을 하나씩 불러와서 np array 형식으로 image_list 에 저장
            for f in files:
                animate_image = AnimateImage(animate=animate, image=f, image_base64="", left_right=l_r, toon_comic=t_c,
                                             ani_effect=ani_effect, transition_effect=tran_effect)
                animate_image.save()
                path = 'media/' + str(animate_image.image)
                with open(path, 'rb') as img:
                    base64_string = base64.b64encode(img.read())
                    tmp = str(base64_string)[2:-1]
                animate_image.image_base64 = tmp
                animate_image.save()
                im = Image.open(f)
                img_array = np.array(im)
                image_list.append(img_array)

            # toon 방식
            if t_c == 'T':
                # 전처리
                labels_cut, labels_bubble = image_preproc(image_list)
                # 컷 분리
                bubbles, centroids, bubble_centers = make_cut_bubble(image_list, labels_bubble, l_r, is_bubble=True)
                cuts, centroids_cut, polygons = make_cut_bubble(image_list, labels_cut, l_r, is_bubble=False)
                # 객체 생성과 동시에 말풍선과 컷을 매칭합니다.
                FrameBook = ComicFrameBook(ani_effect, bubbles, cuts, polygons, bubble_centers,
                                           page_len=len(image_list))
                # ==================================================수정=========================================
                text_bubble_len_list = FrameBook.txt_per_bub_list
                img_list = FrameBook.makeframe_proc()

            # comic 책 방식
            else:
                img_list, text_bubble_len_list = make_page_cut(image_list, l_r)

            # 반환값은 저장된 영상위치
            video_path = view_seconds(img_list, t_c, ani_effect, tran_effect, text_bubble_len_list)
            # =================================================================================================

            # gpu session 비워주기
            tf.keras.backend.clear_session()
            gc.collect()
            # 다시 model에 저장
            animate = form.save(commit=False)
            animate.ani = video_path
            animate.save()

            return HttpResponseRedirect(reverse('animateapp:detail', kwargs={'pk': animate.pk}))
        else:
            return self.form_invalid(animate)


class AnimateDetailView(DetailView):
    model = Animate
    context_object_name = 'target_animate'
    template_name = 'animateapp/detail.html'
