import torch.nn as nn




def make_divisible(v, divisor=1, min_value=1):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class RWLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, us=[True, True]):
        in_features_max = in_features
        out_features_max = out_features
        max_width = 1.0
        if us[0]:
            in_features_max = make_divisible(
                in_features * max_width)
        if us[1]:
            out_features_max = make_divisible(
                out_features * max_width)
        super(RWLinear, self).__init__(
            in_features_max, out_features_max, bias=bias)
        self.in_features_basic = in_features
        self.out_features_basic = out_features
        self.width_mult = 1.0
        self.us = us

    def forward(self, input):
        print("self.width_mult is "+str(self.width_mult))
        in_features = self.in_features_basic
        out_features = self.out_features_basic
        if self.us[0]:
            in_features = make_divisible(
                self.in_features_basic * self.width_mult)
        if self.us[1]:
            out_features = make_divisible(
                self.out_features_basic * self.width_mult)
        weight = self.weight[:out_features, :in_features]
        if self.bias is not None:
            bias = self.bias[:out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)