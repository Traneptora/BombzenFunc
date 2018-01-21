import vapoursynth as vs
import edi_rpow2
import chromeo
import kagefunc as kageru
import mvsfunc as mvf
import havsfunc as haf
from math import factorial

core = vs.core

### Plane Manipulation ###

# Extract an slice from a plane-array based on a list of indices
def slice_plane_array(plane_array, planes=None):
    if planes == None:
        return plane_array[:]
    else:
        return [plane_array[n] for n in planes]

# Turn a clip into a plane array of GRAY, with an optinal list of indices
def clip_to_plane_array(clip: vs.VideoNode, planes=None):
    return [core.std.ShufflePlanes(clip, x, colorfamily=vs.GRAY) for x in (range(clip.format.num_planes) if planes == None else planes)]

# Return a the first plane of a non-RGB clip
def getY(clip: vs.VideoNode) -> vs.VideoNode:
    if clip.format.color_family == vs.RGB:
        raise ValueError('Convert to YUV or equivalent first!')
    return clip_to_plane_array(clip, planes=[0])[0];

# Convert a plane array back to a clip
# Optionally with a list of indices
# Color family is automatically guessed based on number of planes:
#  - 1 plane = GRAY
#  - 3 planes = YUV
#  - other = wtf????
def plane_array_to_clip(plane_array, planes=None, colorfamily=None) -> vs.VideoNode:
    plane_array = slice_plane_array(plane_array, planes)
    if colorfamily == None:
        if len(plane_array) == 1:
            colorfamily = vs.GRAY
        elif len(plane_array) == 3:
            colorfamily = vs.YUV
        else:
            raise ValueError('Not one or three planes?')
    return core.std.ShufflePlanes(clips=plane_array, planes=[0] * len(plane_array), colorfamily=colorfamily)

# Ensure the input is a plane array. Can be passed a clip or an array
# Optionally sliced with a list of indices
# If a plane array is passed then the returned array is a shallow copy
# It never returns a reference to the original
def as_plane_array(clip_or_plane_array, planes=None):
    if isinstance(clip_or_plane_array, vs.VideoNode):
        return clip_to_plane_array(clip=clip_or_plane_array, planes=planes)
    else:
        return clip_or_plane_array[:] if planes == None else [clip_or_plane_array[n] for n in planes]

# Ensure the input is a clip. Can be passed a clip or a plane array
# If passed a clip, a reference back is returned
# Else the plane array is converted to a clip
# Optionally with a list of indices
def as_clip(clip_or_plane_array, planes=None) -> vs.VideoNode:
    if isinstance(clip_or_plane_array, vs.VideoNode):
        return clip_or_plane_array
    else:
        return plane_array_to_clip(clip_or_plane_array, planes=planes)

# This function essentially replaces planes from old_planes with planes from new_planes
# The function's parsing is fairly robust and you usually don't need to select the frame
#   indices manually.
# If you pass it two planes, for example, it replaces the chroma planes of old_planes
# If you pass it one plane, it replaces the luma plane of old_planes.
# If you pass it three planes, it replaces everything in old_planes.
# If you give it a list of indices then it will replace exactly those planes. So e.g.
#   plane_replace(old_planes, new_planes, planes=[1, 2]) replaces the U and V planes of
#   old_planes with the U and V planes of new_planes.
# planeshift represents the relative difference in indices between old_planes and new_planes.
#   e.g. If old_planes is three planes and new_planes is two planes, then planeshift defaults to 1, since
#   the goal is to replace planes 1 and 2 of old_planes with planes 0 and 1 of new_planes.
def plane_replace(old_planes, new_planes, planes=None, planeshift=None) -> vs.VideoNode:
    wanted_planes = as_plane_array(old_planes)
    new_planes = as_plane_array(new_planes)
    if planes == None:
        planes = [n for n in range(len(new_planes))]
    # Assume [1, 2], i.e. chroma, if new_planes has length 2 and planes=None
    if planeshift == None:
        planeshift = 1 if len(new_planes) == 2 else 0
    for n in planes:
        wanted_planes[n:n+1] = [new_planes[n-planeshift]]
    return as_clip(wanted_planes)

### Format Stuff ###

# Conveneince Wrapper
def get_format(clip_or_format) -> vs.Format:
    if isinstance(clip_or_format, vs.VideoNode):
        return clip_or_format.format
    elif isinstance(clip_or_format, vs.Format):
        return clip_or_format
    else:
        raise ValueError('Invalid clip or format')

# Slightly more convenient format.replace()
# since we can pass it a clip
def create_new_format(clip_or_format, colorfamily=None, bits=None, subsampling=None) -> vs.Format:
    src = get_format(clip_or_format)
    colorfamily = src.color_family if colorfamily == None else colorfamily
    bits = src.bits if bits == None else bits
    subsampling_w = src.subsampling_w if subsampling == None else subsampling
    subsampling_h = src.subsampling_h if subsampling == None else subsampling
    sample_type = vs.INTEGER if bits < 32 else vs.FLOAT
    return core.register_format(colorfamily, sample_type, bits, subsampling_w, subsampling_h)

# Maximum value of the format (or the format of the clip) passed.
# Useful for scaling parameters to be 0-255 independent of the internal pixel format
def max_of_format(clip_or_format, chroma_shift=False):
    pixel_format = get_format(clip_or_format)
    if pixel_format.sample_type == vs.FLOAT:
        if chroma_shift:
            return 0.5
        else:
            return 1.0
    else:
        if chroma_shift:
            return 2**(pixel_format.bits_per_sample-1) - 1
        else:
            return 2**pixel_format.bits_per_sample - 1

# Based on Frechdachs's Depth
def depth(src: vs.VideoNode, bits, dither_type=None, range=None, range_in=None) -> vs.VideoNode:
    if src.format.bits_per_sample == bits and range == range_in:
        return src
    if dither_type == None:
        if src_bits < bits or bits >= 16:
            dither_type = 'none'
        else:
            dither_type = 'error_diffusion'
    dest_format = create_new_format(src, bits=bits)
    return src.resize.Point(format=dest_format, dither_type=dither_type, range=range, range_in=range_in)

### Wrappers ###

# Nnedi3 Rpow2, but with znedi3
def znedi3_rpow2(clip: vs.VideoNode,rfactor=2,correct_shift="zimg",nsize=4,nns=4,qual=2,etype=None,pscrn=4,exp=2,opt=None,int16_prescreener=None,int16_predictor=None) -> vs.VideoNode:
    def edi(clip,field,dh):
        return core.znedi3.nnedi3(clip=clip,field=field,dh=dh,nsize=nsize,nns=nns,qual=qual,etype=etype,pscrn=pscrn,opt=opt,int16_prescreener=int16_prescreener,int16_predictor=int16_predictor,exp=exp)
    return edi_rpow2.edi_rpow2(clip=clip,rfactor=rfactor,correct_shift=correct_shift,edi=edi)

# This is just a wrapper to chromeo.ReconstructEx
# Extract the luma from lumaclip and the chroma from chromaclip
# and then feed it to Duplex's Chroma Reconstruction Filter
def reconstruct_ex(lumaclip: vs.VideoNode, chromaclip: vs.VideoNode) -> vs.VideoNode:
    luma, chroma_u, chroma_v = clip_to_plane_array(chromaclip)
    return chromeo.ReconstructEx(getY(lumaclip), chroma_u, chroma_v, slow=True)

# Downscale the Luma to the specificed size using Spline36
# then use Duplex's Chroma Reconstruction Filter
# to reconstruct the chroma upward to 4:4:4
def spline36_reconstruct(clip: vs.VideoNode, width=None, height=None) -> vs.VideoNode:
    if width == None:
        width = clip.width
    if height == None:
        height = clip.height
    luma = getY(clip)
    if width != luma.width or height != luma.height:
        luma = luma.resize.Spline36(width=width, height=height)
    return reconstruct_ex(luma, clip)

# Downscale the Luma to the specificed size using DebilinearM
# then use Duplex's Chroma Reconstruction Filter
# to reconstruct the chroma upward to 4:4:4
def debilinear_reconstruct(clip: vs.VideoNode, width=None, height=None, mask_detail=True, mask_threshold=0.05, show_mask=False) -> vs.VideoNode:
    if width == None:
        width = clip.width
    if height == None:
        height = clip.height
    luma = getY(clip)
    if width != luma.width or height != luma.height:
        luma = kageru.inverse_scale(luma, width=width, height=height, mask_detail=mask_detail, mask_threshold=mask_threshold, show_mask=show_mask)
    return reconstruct_ex(luma, clip)

# Descale the luma with debilinear
# and downscale or upscale the chroma with Lanczos
# default is 4:4:4 but you can set chroma_width and chroma_height to be half as well
def debilinear_lanczos(clip: vs.VideoNode, width=None, height=None, scale_to_444=True, mask_detail=True, mask_threshold=0.05, show_mask=False) -> vs.VideoNode:
	if width == None:
        width = clip.width
    if height == None:
        height = clip.height
    chroma_width = width if scale_to_444 else width // 2
    chroma_height = height if scale_to_444 else height // 2
    planes = clip_to_plane_array(clip)
    luma_plane = kageru.inverse_scale(planes[0], width=width, height=height, mask_detail=mask_detail, mask_threshold=mask_threshold, show_mask=show_mask)
    if mask_detail and show_mask:
        return luma_plane
    chroma_planes = [plane.resize.Lanczos(width=chroma_width, height=chroma_height) for plane in planes[1:]]
    return plane_replace(luma_plane, chroma_planes, planeshift=1)

# requires 32-bit precision
# noise_level above 1 is highly discouraged
# Caffe is shaper but modifies color and can introduce aliasing
# w2x is color-accurate but smudges more
def waifu2x_denoise(clip: vs.VideoNode, noise_level=1, scale_ratio=1, photo=False, caffe=False) -> vs.VideoNode:
    src_fmt = clip.format
    clip = mvf.ToRGB(clip)
    if caffe:
        clip = clip.caffe.Waifu2x(noise=noise_level, scale=scale_ratio, model=(4 if photo else 3) if (scale_ratio > 1) else (2 if photo else 1))
    else:
        clip = clip.w2xc.Waifu2x(noise=noise_level, scale=scale_ratio, photo=photo)
    if src_fmt.color_family == vs.YUV:
        return mvf.ToYUV(clip)
    elif src_fmt.color_family == vs.GRAY:
        return getY(mvf.ToYUV(clip))
    else:
        return clip

# Binarize on a scale of 0-255
def binarize(clip: vs.VideoNode, threshold=127) -> vs.VideoNode:
    return clip.std.Binarize(threshold=(threshold * max_of_format(clip) / 255))

# Plane-friendly bilateral wrapper, using the luma as a reference for all planes
def bilateral(clip: vs.VideoNode, planes=None) -> vs.VideoNode:
    luma = getY(clip)
    filtered_planes = [core.bilateral.Bilateral(plane, luma) for plane in clip_to_plane_array(clip, planes)]
    return plane_replace(clip, filtered_planes, planes)

### My Own Stuff ###

def _remove_grain_vertical(clip: vs.VideoNode) -> vs.VideoNode:
    return clip.rgvs.VerticalCleaner(mode=2).std.Transpose().rgvs.VerticalCleaner(mode=2).std.Transpose()

def _remove_grain_horizontal(clip: vs.VideoNode) -> vs.VideoNode:
    return clip.std.Transpose().rgvs.VerticalCleaner(mode=2).std.Transpose().rgvs.VerticalCleaner(mode=2)

# A version of core.rgvs.VerticalCleaner(mode=2) that runs in a significantly more symmetrical manner
# It is a bit slower though, but shouldn't be *that* much slower.
def remove_grain(clip: vs.VideoNode) -> vs.VideoNode:
    return core.std.Expr(clips=[_remove_grain_vertical(clip), _remove_grain_horizontal(clip)], expr='x y max')

# Standard mathematical binomial function
def binomial(n, k):
    return factorial(n) // (factorial(k) * factorial(n-k))

# Returns the nth row of pascal's triangle
# Used to construct a gaussian blur convolution
def pascal_vector(n):
    return [binomial(n, k) for k in range(0, n+1)]

# Plane-friendly gaussian blur
# matrix_size must be odd, usually 3, 5, or 7
def gaussian_blur(clip: vs.VideoNode, matrix_size=5, planes=None):
    gaussian_vector = pascal_vector(matrix_size - 1)
    # A Gaussian Blur is a separable convolution
    filtered_planes = [plane.std.Convolution(matrix=gaussian_vector, mode='v') for plane in clip_to_plane_array(clip, planes)]
    filtered_planes = [plane.std.Convolution(matrix=gaussian_vector, mode='h') for plane in filtered_planes]
    return plane_replace(clip, filtered_planes, planes)

# Dehalo a gray clip
def _dehalo0(clip: vs.VideoNode, dehalo_radius, mask, darkstr, brightstr) -> vs.VideoNode:
    dehalo_clip = haf.DeHalo_alpha(clip, rx=dehalo_radius, ry=dehalo_radius, darkstr=darkstr, brightstr=brightstr)
    if mask == None:
        return dehalo_clip
    else:
    	# In case there's chroma subsampling we have to Spline36 it
        return core.std.MaskedMerge(clip, dehalo_clip, mask.resize.Spline36(width=clip.width, height=clip.height))

# Plane-selection-friendly masking dehalo wrapper
def dehalo(clip: vs.VideoNode, mask_radius=2, dehalo_radius=2, show_mask=False, planes=None, darkstr=0.0, brightstr=0.6) -> vs.VideoNode:
    if mask_radius == None:
        mask = None
    else:
        mask = dhh.mask(getY(clip), dha=True, smooth=True, radius=mask_radius)
    if show_mask:
        return mask
    filtered_planes = [_dehalo0(plane, dehalo_radius=dehalo_radius, mask=mask, darkstr=darkstr, brightstr=brightstr) for plane in clip_to_plane_array(clip, planes)]
    return plane_replace(clip, filtered_planes, planes=planes)


# Light sharpening filter
# I wrote this one myself, so no promises that it's any good, since I'm basically a noob
# It's based on a combination of an unsharp and a selective gaussian blur to smooth out extra crud created by the sharpen
def sharpen(clip: vs.VideoNode, matrix_size=5, strength=0.35, planes=None):
	# Construct a blurred version of the clip
    blurred = gaussian_blur(clip, matrix_size=matrix_size, planes=planes)
    # Subtract off a fraction of it to create an unsharped clip
    unsharped = core.std.Expr(clips=[clip, blurred], expr='x y - ' + str(strength) + ' * x +')
    # core.std.Expr is not plane-friendly so only take the planes we want to filter
    unsharped = plane_replace(clip, unsharped, planes=planes)
    # Created a blurred version of the previous clip
    blurred_roundtwo = gaussian_blur(unsharped, matrix_size=matrix_size, planes=planes)
    # use mvf.LimitFilter to turn the blurred version into a selective gaussian blur
    # So edges are still preserved but random crud created by the sharpen is gone
    haloclip = mvf.LimitFilter(blurred_roundtwo, unsharped, planes=planes)
    # Unsharp exacerbates halos so this cleans them right up
    # Don't assume you won't have to dehalo because this is here
    dehaloclip = dehalo(haloclip, mask_radius=2, dehalo_radius=2, planes=planes, darkstr=1.0, brightstr=1.0)
    return dehaloclip

# A plane-friendly rewritten custom warpsharp that's better than AWarpSharp2
# It still uses core.warp.AWarp but replaces all the other components itself
# Creation of the mask is outsourced to kagefunc.kirsch()
# The blur radius is far higher than in core.warp.AWarpSharp2 since it's a bombzenfunc.gaussian_blur(), not a BoxBlur and is thus much weaker
# Depth uses units from AWarp, not AWarpSharp2, so depth=3 is the default, not depth=16.
def warpsharp(clip: vs.VideoNode, mask=None, show_mask=False, thresh=48, blur_radius=7, blur_passes=2, depth=3, planes=None) -> vs.VideoNode:
    if mask == None:
        mask = kageru.kirsch(getY(clip))
    mask = binarize(mask, threshold=thresh)
    for i in range(blur_passes):
        mask = gaussian_blur(mask, matrix_size=blur_radius*2-1)
    if show_mask:
        return mask
    filtered_planes = [core.warp.AWarp(clip=plane, mask=mask.resize.Spline36(width=plane.width, height=plane.height), depth=depth) for plane in clip_to_plane_array(clip, planes)]
    return plane_replace(clip, filtered_planes, planes=planes)
