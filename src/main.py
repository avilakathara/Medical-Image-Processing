
from src.segmentation.segmentation import segment
from src.slice_select.optimization import get_optimal_slice
from src.uncertainty.uncertainty import calculate_uncertainty_fields

# load 3D image (proxies)
# TODO

# init user interface
# TODO

# load 3D image from selection
# TODO

isAccepted = False

while not isAccepted:

    # segment (caution: first time and post-user-input may differ)
    segment()

    # calculate uncertainty fields
    calculate_uncertainty_fields()

    # decide if the segmentation is good enough
    # TODO

    if isAccepted:
        break

    # get the most uncertain slice for user input
    get_optimal_slice()

    # present to user and receive changes
    # TODO

