# color features
spatialFeat = False
spatial_clr_arr = ['HSV', 'RGB', 'HLS', 'LUV', 'YUV']
spatial_clr = 'RGB'
spatial = 32
histFeat = False
histbin = 32
# hog features
hogFeat = True
hog_clrspace_arr = ['HSV', 'RGB', 'HLS', 'LUV', 'YUV'] # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
hog_clrspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = [0,1,2] # Can be 0, 1, 2