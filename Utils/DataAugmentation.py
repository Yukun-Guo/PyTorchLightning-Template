import numpy as np
try:
    from scipy.ndimage import zoom as ndi_zoom
except Exception:  # pragma: no cover - if scipy isn't available the user will get a clear error when using resizing
    ndi_zoom = None


class RandomExtractSlice3D(object):
    """Randomly extract a 3D slice from a volume in a sample. volume is a 3D numpy array with channel (height,width,channel)
    The extracted slice is along the channel dimension. extracted slice has shape (height, width,output_channel)
    Args:
        output_channel (int): Desired output size along depth. 
        sample (dict): {'img': img, 'mask': mask}
    """

    def __init__(self, output_channel, training=True):
        self.output_channel = output_channel
        self.training = training

    def __call__(self, sample):
        img, mask = sample["img"], sample["mask"]
        _, _, c = img.shape
        if self.training:
            z = np.random.randint(0, c - self.output_channel)
        else:
            z = (c - self.output_channel) // 2
        
        img = img[:, :, z:z + self.output_channel]
        # mask is only one channel, corresponding to the center slice of the extracted img volume
        mask = mask[:, :, z + self.output_channel // 2]
        return {"img": img, "mask": mask}

        return {"img": img, "mask": mask}

    def __repr__(self):
        return self.__class__.__name__ + "(output_channel={0})".format(
            self.output_channel
        )


class RandomResizingCrop2D(object):
    """Randomly resizing and crop a volume in a sample. volume is a 2D numpy array with channel (height,width,channel)

    Args:
        output_size (tuple or int): Desired output size. If int, cube crop
            is made.
        padding (int or tuple, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a tuple of
            length 3 is provided this is used to pad left, right, top, bottom,
            front, back borders. If a tuple of length 6 is provided this is
            used to pad the left, right, top, bottom, front, back borders
            respectively. Padding with non-constant mode needs the image to be
            padded to the borders if the padding is larger than the image.
            example: (1, 1, 1) or ((1, 1),(1, 1), (1, 1))
        pad_if_needed (bool, optional): It will pad the image if smaller than
            the desired size to avoid raising an exception
        fill (int or tuple or float or str, optional): Pixel fill value for
            constant fill.  default is 0.
        padding_mode (str, optional): Type of padding. Should be: constant,
            edge, reflect or symmetric. Default is constant.
        sample (dict): {'img': img, 'mask1': mask1, 'mask2': mask2}

    """

    def __init__(
        self,
        output_size,  # (height, width)
        padding=None,
        pad_if_needed=True,
        resizing_factor_range=(0.8, 1.2),
        fill=0,
        padding_mode="constant",
        training=True
    ):
        self.size = output_size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        # allowed range for random resizing factor (min, max)
        self.resizing_factor_range = resizing_factor_range
        self.fill = fill
        self.padding_mode = padding_mode
        self.training = training

    def get_params2d(self, img, output_size, mask=None):
        h, w = img.shape[:2]
        th, tw = output_size[:2]

        non_zero_indices = np.nonzero(np.sum(mask, 1))
        min_idx = np.min(non_zero_indices)
        max_idx = np.max(non_zero_indices)
        diff = max_idx - min_idx + 1
        if diff < th:
            range_h = [max(0, min_idx-diff//2), min_idx]
        else:
            range_h = [max(0, min_idx-2), min_idx]

        range_w = 1 if w == tw else w - tw

        i = np.random.randint(range_h[0], range_h[1]+1)
        if i + th > h:
            i = h - th
        j = np.random.randint(0, range_w)
        return i, j, th, tw

    def __call__(self, sample):
        img, mask = sample["img"], sample["mask"]
            
        if self.padding is not None:
            img = np.pad(img, self.padding, self.fill, self.padding_mode)
            mask = np.pad(mask, self.padding, self.fill, self.padding_mode)

        # optionally resize (50% chance)
        if np.random.rand() < 0.5:
            if ndi_zoom is None:
                raise RuntimeError(
                    "scipy.ndimage.zoom is required for RandomResizingCrop2D resizing but scipy is not available"
                )
            factor = float(
                np.random.uniform(self.resizing_factor_range[0], self.resizing_factor_range[1])
            )
            # img expected shape: (H, W,C) or (H, W)
            if img.ndim == 3:
                zoom_factors = (factor, factor,1.0)
                img = ndi_zoom(img, zoom_factors, order=1)
            elif img.ndim == 2:
                img = ndi_zoom(img, (factor, factor), order=1)
                

            # mask: nearest neighbor to keep labels
            if mask.ndim == 2:
                mask = ndi_zoom(mask, (factor, factor), order=0)
            elif mask.ndim == 3:
                zoom_factors = ( factor, factor,1.0)
                mask = ndi_zoom(mask, zoom_factors, order=0)

        # pad the height if needed
        size = img.shape[:2]  # [h, w]
        pad_h = (
            (
                int(np.round((self.size[0] - size[0]) / 2)),
                int(self.size[0] - size[0] -
                    np.round((self.size[0] - size[0]) / 2)),
            )
            if self.pad_if_needed and size[0] < self.size[0]
            else (0, 0)
        )
        pad_w = (
            (
                int(np.round((self.size[1] - size[1]) / 2)),
                int(self.size[1] - size[1] -
                    np.round((self.size[1] - size[1]) / 2)),
            )
            if self.pad_if_needed and size[1] < self.size[1]
            else (0, 0)
        )

        # pad image (channel-first assumed for 3D)
        if img.ndim == 3:
            img = np.pad(
                img,
                (pad_h, pad_w,(0, 0)),
                constant_values=self.fill,
                mode=self.padding_mode,
            )
        else:
            img = np.pad(
                img,
                (pad_h, pad_w),
                constant_values=self.fill,
                mode=self.padding_mode,
            )

        # pad mask
        mask = np.pad(
            mask,
            (pad_h, pad_w),
            constant_values=self.fill,
            mode=self.padding_mode,
        )

        i, j, h, w = self.get_params2d(img, self.size, mask)

        if not self.training:
            i=0
            j=0

        # crop the image and mask to requested size
        if img.ndim == 3:
            img = img[i: i + h, j: j + w, :].copy()
        else:
            img = img[i: i + h, j: j + w].copy()
        mask = mask[i: i + h, j: j + w].copy()

        return {"img": img, "mask": mask}

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(
            self.size, self.padding
        )


class RandomGammaAdjust(object):
    """Randomly change the gamma of the image. the image can be 2D, 3D or 4D.
    Args:
        gamma_range (tuple): range of gamma change.
        sample (dict): {'img': img, 'mask': mask}
    """

    def __init__(self, gamma_range=(0.3, 1.2)):
        self.gamma_range = gamma_range

    def __call__(self, sample):
        img, mask = sample["img"], sample["mask"]
        gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
        img = np.power(img / 255.0, gamma) * 255.0
        img = np.clip(img, 0, 255)
        return {"img": img, "mask": mask}

    def __repr__(self):
        return self.__class__.__name__ + "(gamma_range={0})".format(
            self.gamma_range
        )


class GrayJitter(object):
    """Randomly change the brightness and contrast of the image. the image can be 2D, 3D or 4D.
    Args:
        brightness_range (tuple): range of brightness adjustment (-value to +value).
        contrast_range (tuple): range of contrast change. Values < 1.0 reduce contrast, > 1.0 increase contrast.
        max_value (int): max value of the image.
        p (float): probability of applying the transformation.
        sample (dict): {'img': img, 'mask': mask}
    """

    def __init__(self, brightness_range=(-30, 30), contrast_range=(0.6, 1.4), max_value=255, p=0.5):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.max_value = max_value
        self.p = p

    def __call__(self, sample):
        img, mask = sample["img"], sample["mask"]
        if np.random.rand() < self.p:
            # Apply contrast adjustment
            contrast_scale = np.random.uniform(
                self.contrast_range[0], self.contrast_range[1]
            )
            meanv = np.mean(img)
            img = (img - meanv) * contrast_scale + meanv
            
            # Apply brightness adjustment
            brightness_shift = np.random.uniform(
                self.brightness_range[0], self.brightness_range[1]
            )
            img = img + brightness_shift
            
            img = np.clip(img, 0, self.max_value)
        return {"img": img, "mask": mask}

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(brightness_range={0}, contrast_range={1}, p={2})".format(
                self.brightness_range, self.contrast_range, self.p
            )
        )


class LowerContrast(object):
    """Reduce the contrast of the image by scaling pixel values toward the mean.
    This simulates low-contrast imaging conditions.
    
    Args:
        contrast_factor_range (tuple): range of contrast reduction factor (0.0 to 1.0).
                                       0.0 = all pixels become mean (no contrast),
                                       1.0 = original image (full contrast).
        max_value (int): max value of the image.
        sample (dict): {'img': img, 'mask': mask}
    """

    def __init__(self, contrast_factor_range=(0.5, 0.8), max_value=255):
        self.contrast_factor_range = contrast_factor_range
        self.max_value = max_value

    def __call__(self, sample):
        img, mask = sample["img"], sample["mask"]
        contrast_factor = np.random.uniform(
            self.contrast_factor_range[0], self.contrast_factor_range[1]
        )
        meanv = np.mean(img)
        # Scale toward mean: img_new = mean + (img - mean) * factor
        img = meanv + (img - meanv) * contrast_factor
        img = np.clip(img, 0, self.max_value)
        return {"img": img, "mask": mask}

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(contrast_factor_range={0})".format(self.contrast_factor_range)
        )


class AddGaussianNoise(object):
    """Add Gaussian noise to the image. the image can be 2D, 3D or 4D.

    Args:
        mean (float): mean of the Gaussian distribution.
        std_range (tuple): range for standard deviation of the Gaussian distribution.
        p (float): probability of applying the transformation.
        sample (dict): {'img': img, 'mask': mask}
    """

    def __init__(self, mean=0.0, std_range=(2.0, 10.0), p=0.3):
        self.std_range = std_range
        self.mean = mean
        self.p = p

    def __call__(self, sample):
        img, msk = sample["img"], sample["mask"]
        if np.random.rand() < self.p:
            std = np.random.uniform(self.std_range[0], self.std_range[1])
            noise = np.random.normal(self.mean, std, img.shape)
            img = np.clip(img + noise, 0, 255)
        return {"img": img, "mask": msk}

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std_range={1}, p={2})".format(
            self.mean, self.std_range, self.p
        )


class RandomBlur(object):
    """Apply random blur to simulate out-of-focus or motion blur effects.
    
    Args:
        kernel_size_range (tuple): range of kernel sizes for blur (must be odd numbers).
        p (float): probability of applying the transformation.
        sample (dict): {'img': img, 'mask': mask}
    """

    def __init__(self, kernel_size_range=(3, 7), p=0.3):
        self.kernel_size_range = kernel_size_range
        self.p = p

    def __call__(self, sample):
        img, mask = sample["img"], sample["mask"]
        if np.random.rand() < self.p:
            try:
                from scipy.ndimage import gaussian_filter
                # Random sigma for Gaussian blur
                sigma = np.random.uniform(0.5, 2.0)
                if img.ndim == 3:
                    # Apply blur to each channel
                    for c in range(img.shape[2]):
                        img[:, :, c] = gaussian_filter(img[:, :, c], sigma=sigma)
                else:
                    img = gaussian_filter(img, sigma=sigma)
                img = np.clip(img, 0, 255)
            except ImportError:
                # Fallback to simple averaging if scipy not available
                kernel_size = np.random.choice([3, 5, 7])
                kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
                if img.ndim == 3:
                    for c in range(img.shape[2]):
                        img[:, :, c] = np.clip(
                            np.convolve(img[:, :, c].flatten(), kernel.flatten(), mode='same').reshape(img[:, :, c].shape),
                            0, 255
                        )
                else:
                    img = np.clip(
                        np.convolve(img.flatten(), kernel.flatten(), mode='same').reshape(img.shape),
                        0, 255
                    )
        return {"img": img, "mask": mask}

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(kernel_size_range={0}, p={1})".format(self.kernel_size_range, self.p)
        )


class RandomIntensityScale(object):
    """Randomly scale the intensity values to simulate different imaging conditions.
    
    Args:
        scale_range (tuple): range of scaling factors.
        p (float): probability of applying the transformation.
        sample (dict): {'img': img, 'mask': mask}
    """

    def __init__(self, scale_range=(0.7, 1.3), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, sample):
        img, mask = sample["img"], sample["mask"]
        if np.random.rand() < self.p:
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            img = img * scale
            img = np.clip(img, 0, 255)
        return {"img": img, "mask": mask}

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(scale_range={0}, p={1})".format(self.scale_range, self.p)
        )


class AddNoiseToLabelRegion(object):
    """Blend label regions with surrounding regions to make them look more similar.
    Creates natural-looking transformations by sampling and blending characteristics 
    from neighboring non-label regions.
    
    Args:
        p (float): probability of applying the transformation.
        label_value (list): list of label values to apply transformation to.
        blur_sigma (float): sigma for Gaussian blur to smooth mask boundaries.
        blend_strength (float): strength of blending (0-1), where 1 means full replacement.
        sample (dict): {'img': img, 'mask': mask}
    """

    def __init__(self, p=0.5, label_value:list=[1,2], blur_sigma=5.0, blend_strength=0.3):
        self.p = p
        self.label_value = label_value
        self.blur_sigma = blur_sigma
        self.blend_strength = blend_strength

    def __call__(self, sample):
        img, mask = sample["img"], sample["mask"]
        if np.random.rand() < self.p:
            try:
                from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
                
                # Create mask for specified labels
                label_mask = np.isin(mask, self.label_value)
                
                # Create a surrounding region mask (dilated - eroded)
                struct = np.ones((9, 9))
                dilated = binary_dilation(label_mask, structure=struct, iterations=2)
                surrounding_mask = dilated & ~label_mask
                
                # If no surrounding region exists, skip
                if not np.any(surrounding_mask):
                    return {"img": img, "mask": mask}
                
                # Choose augmentation type
                aug_type = np.random.choice(['blur_blend', 'intensity_match', 'texture_blend'])
                
                if aug_type == 'blur_blend':
                    # Blur the label region to blend with surroundings
                    img_blurred = img.copy()
                    blur_sigma_local = np.random.uniform(2.0, 5.0)
                    
                    if img.ndim == 3:
                        for c in range(img.shape[2]):
                            img_blurred[:, :, c] = gaussian_filter(img[:, :, c], sigma=blur_sigma_local)
                    else:
                        img_blurred = gaussian_filter(img, sigma=blur_sigma_local)
                    
                    # Create smooth transition mask
                    transition_mask = label_mask.astype(np.float32)
                    transition_mask = gaussian_filter(transition_mask, sigma=self.blur_sigma)
                    
                    if img.ndim > mask.ndim:
                        transition_mask = np.expand_dims(transition_mask, axis=-1)
                    
                    # Blend with reduced strength
                    blend_factor = self.blend_strength
                    img = img * (1 - blend_factor * transition_mask) + img_blurred * (blend_factor * transition_mask)
                
                elif aug_type == 'intensity_match':
                    # Match intensity statistics of label region to surrounding region
                    if img.ndim == 3:
                        for c in range(img.shape[2]):
                            # Get statistics from surrounding region
                            surrounding_pixels = img[:, :, c][surrounding_mask]
                            if len(surrounding_pixels) > 0:
                                surr_mean = np.mean(surrounding_pixels)
                                surr_std = np.std(surrounding_pixels)
                                
                                # Get statistics from label region
                                label_pixels = img[:, :, c][label_mask]
                                if len(label_pixels) > 0:
                                    label_mean = np.mean(label_pixels)
                                    label_std = np.std(label_pixels) + 1e-6
                                    
                                    # Create adjusted label region
                                    img_adjusted = img[:, :, c].copy()
                                    img_adjusted[label_mask] = ((img_adjusted[label_mask] - label_mean) / label_std) * surr_std + surr_mean
                                    
                                    # Smooth transition
                                    transition_mask = label_mask.astype(np.float32)
                                    transition_mask = gaussian_filter(transition_mask, sigma=self.blur_sigma)
                                    
                                    # Apply with blending
                                    img[:, :, c] = img[:, :, c] * (1 - self.blend_strength * transition_mask) + \
                                                   img_adjusted * (self.blend_strength * transition_mask)
                    else:
                        # Same for grayscale
                        surrounding_pixels = img[surrounding_mask]
                        if len(surrounding_pixels) > 0:
                            surr_mean = np.mean(surrounding_pixels)
                            surr_std = np.std(surrounding_pixels)
                            
                            label_pixels = img[label_mask]
                            if len(label_pixels) > 0:
                                label_mean = np.mean(label_pixels)
                                label_std = np.std(label_pixels) + 1e-6
                                
                                img_adjusted = img.copy()
                                img_adjusted[label_mask] = ((img_adjusted[label_mask] - label_mean) / label_std) * surr_std + surr_mean
                                
                                transition_mask = label_mask.astype(np.float32)
                                transition_mask = gaussian_filter(transition_mask, sigma=self.blur_sigma)
                                
                                img = img * (1 - self.blend_strength * transition_mask) + \
                                      img_adjusted * (self.blend_strength * transition_mask)
                
                elif aug_type == 'texture_blend':
                    # Add subtle texture from surrounding regions
                    if img.ndim == 3:
                        for c in range(img.shape[2]):
                            # Extract high-frequency component from surrounding region
                            img_smooth = gaussian_filter(img[:, :, c], sigma=2.0)
                            texture = img[:, :, c] - img_smooth
                            
                            # Create smooth transition
                            transition_mask = label_mask.astype(np.float32)
                            transition_mask = gaussian_filter(transition_mask, sigma=self.blur_sigma)
                            
                            # Reduce texture in label region with smooth blending
                            texture_factor = 1.0 - self.blend_strength * 0.5  # Reduce texture by half at most
                            img[:, :, c] = img_smooth + texture * (1 - self.blend_strength * 0.5 * transition_mask)
                    else:
                        img_smooth = gaussian_filter(img, sigma=2.0)
                        texture = img - img_smooth
                        
                        transition_mask = label_mask.astype(np.float32)
                        transition_mask = gaussian_filter(transition_mask, sigma=self.blur_sigma)
                        
                        img = img_smooth + texture * (1 - self.blend_strength * 0.5 * transition_mask)
                
                img = np.clip(img, 0, 255)
                
            except ImportError:
                # Fallback: simply blur the label region slightly
                pass
                
        return {"img": img, "mask": mask}

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(mean={0}, std_range={1}, p={2}, label_value={3})".format(
                self.mean, self.std_range, self.p, self.label_value
            )
        )

class ElasticTransform(object):
    """Apply elastic deformation to simulate tissue deformation in medical images.
    
    Args:
        alpha_range (tuple): range for deformation intensity.
        sigma (float): Gaussian filter parameter for smoothing the deformation.
        p (float): probability of applying the transformation.
        sample (dict): {'img': img, 'mask': mask}
    """

    def __init__(self, alpha_range=(30, 40), sigma=5, p=0.3):
        self.alpha_range = alpha_range
        self.sigma = sigma
        self.p = p

    def __call__(self, sample):
        img, mask = sample["img"], sample["mask"]
        if np.random.rand() < self.p:
            try:
                from scipy.ndimage import gaussian_filter, map_coordinates
                
                alpha = np.random.uniform(self.alpha_range[0], self.alpha_range[1])
                shape = img.shape[:2]
                
                # Generate random displacement fields
                dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * alpha
                dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * alpha
                
                # Create meshgrid
                x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
                indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
                
                # Apply transformation
                if img.ndim == 3:
                    img_transformed = np.zeros_like(img)
                    for c in range(img.shape[2]):
                        img_transformed[:, :, c] = map_coordinates(
                            img[:, :, c], indices, order=1, mode='reflect'
                        ).reshape(shape)
                    img = img_transformed
                else:
                    img = map_coordinates(img, indices, order=1, mode='reflect').reshape(shape)
                
                # Apply same transformation to mask
                mask = map_coordinates(mask, indices, order=0, mode='reflect').reshape(shape)
                
                img = np.clip(img, 0, 255)
            except ImportError:
                # Skip if scipy not available
                pass
        return {"img": img, "mask": mask}

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(alpha_range={0}, sigma={1}, p={2})".format(
                self.alpha_range, self.sigma, self.p
            )
        )


class RandomRotate90n(object):
    """Rotate the image by 90, 180, 270 degrees randomly. the image can be 2D or 3D.

    Args:
        axes (tuple): axes to rotate. Default: (0, 1) for 2D, (1, 2) for 3D ?
        sample (dict): {'img': img, 'mask1': mask1, 'mask2': mask2}
    """

    def __init__(self, axes=0):
        self.axes = axes

    def __call__(self, sample):
        image, mask1, mask2 = sample["img"], sample["mask1"], sample["mask2"]
        degree = np.random.randint(0, 3)
        image = np.rot90(image, degree, axes=self.axes)
        mask1 = np.rot90(mask1, degree, axes=self.axes)
        mask2 = np.rot90(mask2, degree, axes=self.axes)
        return {"img": image, "mask1": mask1, "mask2": mask2}


class RandomFlip(object):
    """Horizontally/Vertical flip the given Image randomly with a given probability. the image can be 2D or 3D.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        axis (int): axis to flip. 0 for vertical flip, 1 for horizontal flip. Default value is 0.
        sample (dict): {'img': img, 'mask1': mask1, 'mask2': mask2}
    """

    def __init__(self, p=0.5, axis=0):
        self.p = p
        self.axis = axis

    def __call__(self, sample):
        image, mask = sample["img"], sample["mask"]
        if np.random.random() < self.p:
            image = np.flip(image, axis=self.axis)
            mask = np.flip(mask, axis=self.axis)
        return {"img": image, "mask": mask}


class RandomRoll(object):
    """Randomly roll the image along a given axis. the image can be 2D or 3D.
    Args:
        axis (int): axis to roll. 0 for vertical roll, 1 for horizontal roll. Default value is 0.
        sample (dict): {'img': img, 'mask': mask}
    """

    def __init__(self, axis=0, shift_ratio_range=(0.3, 1)):
        self.axis = axis
        self.shift_ratio_range = shift_ratio_range
    
    def roll_columns(self,img, offsets):
        """
        Roll each column of an image with a different vertical offset (circular wrap).

        Parameters:
            img (np.ndarray): Input image, shape (H, W, C) or (H, W)
            offsets (np.ndarray or list): Array of length W, vertical shift per column
                                        Positive = shift down, negative = shift up

        Returns:
            np.ndarray: Rolled image with same shape as input
        """

        H, W = img.shape[:2]

        # Row indices
        rows = np.arange(H)[:, None]  # shape (H, 1)
        cols = np.arange(W)  # shape (W,)

        if img.ndim == 2:
            rolled = img[(rows - offsets) % H, cols]
        else:
            rolled = img[(rows - offsets) % H, cols, :]

        return rolled
    
    def roll_columns_with_padding(self, img, offsets):
        """
        Roll each column of an image with a different vertical offset,
        padding with zeros instead of circular wrap.

        Parameters:
            img (np.ndarray): Input image, shape (H, W, C) or (H, W)
            offsets (np.ndarray or list): Array of length W, vertical shift per column
                                        Positive = shift down, negative = shift up

        Returns:
            np.ndarray: Rolled image with same shape as input
        """
        img = np.asarray(img)
        offsets = np.asarray(offsets).astype(int)
        H, W = img.shape[:2]
        C = 1 if img.ndim == 2 else img.shape[2]

        # Prepare output array
        rolled = np.zeros_like(img)

        # Create row indices
        rows = np.arange(H)[:, None]  # shape (H,1)
        cols = np.arange(W)  # shape (W,)

        for j, shift in enumerate(offsets):
            if shift > 0:
                # shift down
                src_rows = np.arange(H - shift)
                dst_rows = src_rows + shift
            elif shift < 0:
                # shift up
                src_rows = np.arange(-shift, H)
                dst_rows = np.arange(H + shift)
            else:
                # no shift
                src_rows = np.arange(H)
                dst_rows = src_rows

            if C == 1:
                rolled[dst_rows, j] = img[src_rows, j]
            else:
                rolled[dst_rows, j, :] = img[src_rows, j, :]

        return rolled

    def __call__(self, sample):
        image, mask = sample["img"], sample["mask"]
        if np.random.rand() < 0.33:
            # shift_offsets is a array with colums equal to the shift value along the given axis, the rolling for each colum is different
            shift_ratio = np.random.uniform(
                self.shift_ratio_range[0], self.shift_ratio_range[1]
            )
            shift_offsets = np.floor(np.asarray(range(0, image.shape[self.axis]))*shift_ratio)

            if np.random.rand() < 0.5:
                # reverse the orders
                shift_offsets = shift_offsets[::-1]

            image = self.roll_columns_with_padding(image, shift_offsets)
            mask = self.roll_columns_with_padding(mask, shift_offsets)

        return {"img": image, "mask": mask}



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Functions from before ---

    def roll_columns_with_padding(img, offsets):
        img = np.asarray(img)
        offsets = np.asarray(offsets)
        H, W = img.shape[:2]
        C = 1 if img.ndim == 2 else img.shape[2]
        rolled = np.zeros_like(img)
        rows = np.arange(H)[:, None]
        cols = np.arange(W)
        for j, shift in enumerate(offsets):
            if shift > 0:
                src_rows = np.arange(H - shift)
                dst_rows = src_rows + shift
            elif shift < 0:
                src_rows = np.arange(-shift, H)
                dst_rows = np.arange(H + shift)
            else:
                src_rows = np.arange(H)
                dst_rows = src_rows
            if C == 1:
                rolled[dst_rows, j] = img[src_rows, j]
            else:
                rolled[dst_rows, j, :] = img[src_rows, j, :]
        return rolled

    def roll_columns(img, offsets):
        img = np.asarray(img)
        offsets = np.asarray(offsets)
        H, W = img.shape[:2]
        rows = np.arange(H)[:, None]
        cols = np.arange(W)
        if img.ndim == 2:
            rolled = img[(rows - offsets) % H, cols]
        else:
            rolled = img[(rows - offsets) % H, cols, :]
        return rolled

    # --- Generate a gradient image for visualization ---
    H, W = 100, 200
    # Gradient along horizontal direction (columns)
    gradient = np.tile(np.linspace(0, 255, W, dtype=np.uint8), (H, 1))
    # Convert to RGB for multi-channel test
    img_rgb = np.stack([gradient, gradient, gradient], axis=-1)

    # --- Offsets for each column ---
    offsets = np.linspace(0, 20, W, dtype=int)  # gradually increasing shift
    offsets = -offsets  # reverse for variety
    # --- Apply the functions ---
    rolled_zero = roll_columns_with_padding(img_rgb, offsets)
    rolled_circular = roll_columns(img_rgb, offsets)

    # --- Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
