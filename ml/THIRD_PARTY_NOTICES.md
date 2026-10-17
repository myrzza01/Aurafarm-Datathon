# Источники модели

Визуальная основа: `timm/mobilenetv3_small_100.lamb_in1k`.
Авторская карточка: https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k
В карточке указана лицензия Apache-2.0 (проверено 17.10.2026).
В поставляемых baseline и temporal checkpoint визуальный энкодер заморожен;
классификаторы обучены на локальных данных хакатона отдельно в данном проекте.

Код timm: https://github.com/huggingface/pytorch-image-models
Лицензия Apache-2.0 сохранена в `licenses/timm-LICENSE.txt`.
Код TorchVision: https://github.com/pytorch/vision
Лицензия BSD-3-Clause: https://github.com/pytorch/vision/blob/v0.22.1/LICENSE
Лицензии остальных установленных зависимостей поставляются в соответствующих
пакетах окружения. Не удаляйте их при распространении бинарных зависимостей.

Требования организаторов к видеоданным сохраняются независимо от лицензии
библиотек и исходной предобученной модели.
