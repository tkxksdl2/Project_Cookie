from django.forms import ModelForm, HiddenInput

from animateapp.models import Animate


class AnimateCreationForm(ModelForm):
    class Meta:
        model = Animate
        fields = ['image', 'ani', 'left_right']
        widgets = {
            # 생성시 안보이게 처리
            'ani': HiddenInput(),
        }

