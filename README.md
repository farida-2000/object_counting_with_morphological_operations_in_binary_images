# object_counting_with_morphological_operations_in_binary_images
 This project is about object counting with morphological operations on binary images. First, we convert 
images to grayscale. Using utso method, we define a threshold and convert grayscale image to binary 
(0,1) image. We try some different kernels with different patterns to determine which one is the most 
appropriate for the purpose of dilation and erosion. After applying dilation, we apply erosion with one 
iteration. Using connected components OpenCV function, we try to do the labeling technique and 
present objects’ count. To find the centers of objects, we apply findcontours OpenCV function and find 
the contours, We iterate over the contours and calculate the center in a for loop and write the final 
image in result.png
Applying different patterns of kernels let us find out what is the best kernel to use for 
either erosion or dilation. Labeling technique with connected components helps us to count objects.
