def codeword_threshold(x):
    # torch tensor: > 0.5 = 1; <= 0.5 = 0
    x[x > 0.5] = 1
    x[x <= 0.5] = 0
    return x

def find_index(all_array, element):
    all_array = all_array.tolist()
    element = element.tolist()
    if element in all_array:
        return all_array.index(element)