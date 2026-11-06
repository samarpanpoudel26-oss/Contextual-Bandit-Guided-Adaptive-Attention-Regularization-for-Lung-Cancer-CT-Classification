import tensorflow as tf

# Small constant used to prevent division by zero during normalization and loss calculation
EPSILON = 1e-8


def compute_raw_attention_map(feature_maps):
    """
    Average the channels of the final convolutional
    feature maps to create a spatial attention map.

    Input:  (batch, height, width, channels)
    Output: (batch, height, width, 1)
    """
    # Convert feature maps to float32 for consistent numerical computation
    feature_maps = tf.convert_to_tensor(
        feature_maps, dtype=tf.float32
    )

    # Average feature activations across channels to obtain
    # one attention value for each spatial location
    attention_map = tf.reduce_mean(
        feature_maps,
        axis=-1,
        keepdims=True
    )

    return attention_map


def resize_attention_map(
    attention_map,
    target_size=(256, 256)
):
    """Resize attention maps to the image/mask resolution."""
    # Bilinear interpolation is used to resize the attention map
    # while maintaining a smooth spatial representation
    return tf.image.resize(
        attention_map,
        size=target_size,
        method="bilinear"
    )


def normalize_attention_map(
    attention_map,
    eps=EPSILON
):
    """Apply min-max normalization independently per image."""
    # Get the number of images in the batch
    batch_size = tf.shape(attention_map)[0]

    # Flatten each attention map to calculate its minimum and maximum values
    flattened = tf.reshape(
        attention_map,
        [batch_size, -1]
    )

    # Find the minimum and maximum activation values for each image separately
    minimum = tf.reduce_min(
        flattened, axis=1, keepdims=True
    )

    maximum = tf.reduce_max(
        flattened, axis=1, keepdims=True
    )

    # Apply min-max normalization so that attention values
    # are scaled approximately between 0 and 1
    normalized = (
        flattened - minimum
    ) / (
        maximum - minimum + eps
    )

    # Restore the original spatial dimensions of the attention maps
    return tf.reshape(
        normalized,
        tf.shape(attention_map)
    )


def compute_attention_map(
    feature_maps,
    target_size=(256, 256)
):
    """Create, resize, and normalize the attention map."""
    # Generate the initial spatial attention map from feature activations
    attention_map = compute_raw_attention_map(feature_maps)

    # Resize the attention map to match the CT image and mask resolution
    attention_map = resize_attention_map(
        attention_map,
        target_size=target_size
    )

    # Normalize each attention map independently
    attention_map = normalize_attention_map(attention_map)

    return attention_map


def compute_agal_loss(
    attention_map,
    anatomical_mask,
    eps=EPSILON
):
    """
    Calculate the proportion of attention outside
    the binary anatomical mask.

    Mask convention:
        1 = inside the permitted region
        0 = outside the permitted region
    """
    # Convert both inputs to float32 for mathematical operations
    attention_map = tf.convert_to_tensor(
        attention_map, dtype=tf.float32
    )

    anatomical_mask = tf.convert_to_tensor(
        anatomical_mask, dtype=tf.float32
    )

    # Resize the mask to the attention-map dimensions.
    # This does not correct an incorrectly aligned mask.
    anatomical_mask = tf.image.resize(
        anatomical_mask,
        size=tf.shape(attention_map)[1:3],
        method="nearest"
    )

    # Ensure that the mask is binary.
    # Values >= 0.5 are treated as inside the permitted region (1)
    # Values < 0.5 are treated as outside the permitted region (0)
    anatomical_mask = tf.cast(
        anatomical_mask >= 0.5,
        tf.float32
    )

    # Invert the mask to identify pixels outside the permitted region
    outside_mask = 1.0 - anatomical_mask

    # Retain attention values only at locations outside the mask
    outside_attention = (
        attention_map * outside_mask
    )

    # Calculate the total attention outside the permitted region for each image
    outside_sum = tf.reduce_sum(
        outside_attention,
        axis=[1, 2, 3]
    )

    # Calculate the total attention across the entire image for each image
    total_sum = tf.reduce_sum(
        attention_map,
        axis=[1, 2, 3]
    )

    # Calculate the proportion of attention outside the anatomical region
    # A lower value indicates less attention outside the permitted region
    outside_ratio = outside_sum / (total_sum + eps)

    # Average the outside-attention ratio across all images in the batch
    return tf.reduce_mean(outside_ratio)
