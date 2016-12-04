# Ascii Art
This is basically a re-implimentation of the following academic paper on a
method of generating ascii art by converting the image, and the text into
line segments, jigglying them randomly until they best match the ascii art.
[https://pdfs.semanticscholar.org/d42c/069253c40a4795b51afffe53d02ff7f844cd.pdf]
They use a cool method of comparing the ascii art with the original image
using multiple polar samples across cells of the image.

# Our Method
1. Render each character of the font at the target resolution
2. Compute the skeleton of each character
3. Convert the skeleton into a line vector
4. Detect the edges in the input image
5. Convert the edges into vectors
6. Split the image into cells corresponding with an ascii character
7. Find the character which has the least difference with each cell
8. Randomly perturb an line segment
9. Compare the new image representation with the previous, if it is similar enough, keep it, if not, then randomly keep it
10. Repeat back to 7 (Though only one or two cells need be recomputed) until the overall difference between all matched characters is below some threshold, or the algorithm has run a certain number of times
11. Return the resulting ascii characters

