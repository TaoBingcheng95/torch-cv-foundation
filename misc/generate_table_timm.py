import timm
from tqdm import tqdm


def check_features_and_reduction(model_name):
    encoder = timm.create_model(model_name, features_only=True, pretrained=False)
    if not encoder.feature_info.reduction() == [2, 4, 8, 16, 32]:
        raise ValueError

def has_dilation_support(model_name):
    try:
        timm.create_model(model_name, features_only=True, output_stride=8, pretrained=False)
        timm.create_model(model_name, features_only=True, output_stride=16, pretrained=False)
        return True
    except Exception as e:
        print(e)
        return False

def make_table(models):
    model_names = models.keys() #supported.keys()
    max_len1 = max([len(x) for x in model_names]) + 2
    max_len2 = len("support dilation") + 2
    
    l1 = "+" + "-" * max_len1 + "+" + "-" * max_len2 + "+\n"
    l2 = "+" + "=" * max_len1 + "+" + "=" * max_len2 + "+\n"
    top = "| " + "Encoder name".ljust(max_len1 - 2) + " | " + "Support dilation".center(max_len2 - 2) + " |\n"
    
    model_table = l1 + top + l2
    
    for k in sorted(models.keys()):
        support = "✅".center(max_len2 - 3) if models[k]["has_dilation"] else " ".center(max_len2 - 2)
        model_table += "| " + k.ljust(max_len1 - 2) + " | " + support + " |\n"
        model_table += l1
    
    return model_table
    

if __name__ == "__main__":

    supported_models = {}

    with tqdm(timm.list_models()) as names:
        for name in names:
            try:
                check_features_and_reduction(name)
                has_dilation = has_dilation_support(name)
                supported_models[name] = dict(has_dilation=has_dilation)
            except Exception as e:
                print(e)
                continue

    table = make_table(supported_models)
    print(table)
    print(f"Total encoders: {len(supported_models.keys())}")
