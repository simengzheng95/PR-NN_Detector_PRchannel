# find one idx for the matched sequence
def find_index(all_array, element):
    all_array = all_array.tolist()
    element = element.tolist()
    if element in all_array:
        return all_array.index(element)