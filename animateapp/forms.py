from django import forms
from django.forms import ModelForm, HiddenInput

from animateapp.models import Animate


class AnimateForm(ModelForm):
    class Meta:
        model = Animate
        fields = ['image', 'ani', 'left_right']
        widgets = {
            # 생성시 안보이게 처리
            'ani': HiddenInput(),
            #다중이미지 업로드하게 multiple요소 추가
            'image': forms.ClearableFileInput(attrs={'multiple': True})
        }
        labels = {
            #생성시 라벨 변경
            'left_right': '만화책 방향',
        }

