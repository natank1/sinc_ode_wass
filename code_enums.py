from enum import Enum

# class mode_input_data (Enum):
#     fft_magnitude = 1
#     ivector=2
#     combine_data =3

class mode_target (Enum):
    # bayes_target = 4  N/A right now (though interesting study
    bhat = 1
    hellinger = 2
    wassertein = 3

class mode_type (Enum):
    # bayes_target = 4  N/A right now (though interesting study
    resnet = 1
    ivector = 2
    gru = 3
