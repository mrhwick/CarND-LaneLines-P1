# Finding Lane Lines on the Road

### Reflection

### 1. Describe your pipeline.

#### Pipeline Steps

My pipeline consists of 9 steps.

1. Create a copy of the input image for processing into a line overlay image.
2. Convert the image into [HLS Color Space](https://en.wikipedia.org/wiki/HSL_and_HSV) (Choosing this color space rather than grayscale can help with detections of non-white lane lines). ![image](https://cloud.githubusercontent.com/assets/865759/24082341/72019bde-0c9a-11e7-922f-a85382769413.png)
3. Normalize the image values once in HLS. ![image](https://cloud.githubusercontent.com/assets/865759/24082350/84b992fe-0c9a-11e7-9d15-babf6dbc90f5.png)
4. Apply a Gaussian blurring filter with a kernel size of 5. This smoothes out the smaller edges to make them less detectable. ![image](https://cloud.githubusercontent.com/assets/865759/24082391/3fccd90c-0c9b-11e7-859f-9cdff04d748e.png)
5. Apply a Canny edge detection filter with a low threshold of 50 and a high threshold of 110. ![image](https://cloud.githubusercontent.com/assets/865759/24082396/50a4037c-0c9b-11e7-823c-ada13d3f89a8.png)
6. Crop the edge-points image down to a specific region of interest. I chose to make this region of interest a shape that would remove any edge points at or above the horizon, far to the left or right, and on the ground immediately in front of the vehicle. ![image](https://cloud.githubusercontent.com/assets/865759/24082402/5dabec24-0c9b-11e7-9d1e-63acc8616589.png)
7. Apply a Hough transform on the cropped edge points image. Here I chose to opt for discarding shorter lines (any below 40 px), and allowing the larger detected lines to be merged over a gap distance of a 25 px. The threshold parameter was the setting which I tweaked most in training, and settled on a value in the 130-170 votes range. ![image](https://cloud.githubusercontent.com/assets/865759/24082410/7b9e7706-0c9b-11e7-8007-87cffc169a82.png)
8. Use the returned Hough line list to actually generate a line image for the left and right lanes with a linear fit for each line group. The interpretation of the Hough line list for rendering is of special interest, and I'll describe the operation of the `generate_line_image` function below. ![image](https://cloud.githubusercontent.com/assets/865759/24082415/8d6af7d4-0c9b-11e7-8a7e-acfd98626871.png)
9. With the generated image of linear-fit left and right lane lines, we then must only combine the original input image with the generated lane line image. ![image](https://cloud.githubusercontent.com/assets/865759/24082418/9925bfa0-0c9b-11e7-949c-7b05cc7edc1e.png)

#### Generating Linear-fit Left and Right Lane Lines

The first challenge in line processing is determining which lines belong to the left group and which belong to the right group. I accomplished this task by calculating the slope of each line in turn. The line belongs to the left group if the slope is negative, because in computer graphics we are always working in quadrant 4 of the cartesian coordinate system. If the slope is positive, it must belong to the right group. At this step, I also eliminate any lines which do not have a sufficiently extreme slope to be a close match to lane line geometry (any slope with absolute value below 0.5).

The second challenge in line processing is to extrapolate the grouped lines into a single line representative of all the lines belonging to each group. My first inclination was to determine some known values that I would work from to generate these line segments. Some obvious known values were the y values for the top and bottom endpoints of the segments. The top y value would be the middle of the image and the bottom of the image would be the bottom y value. The problem then becomes an exercise of finding the correct x values for each point of each line segment.

In order to find the correct x values for each point, I decided to develop a function `f(y) = x` which would be a linearly fit function to the point data given by each group. This actually simplified my implementation greatly, as I only needed to collect the x and y coords for each group as preprocessing of the line groups. For the actual function generation, I opted to use numpy's [polyfit](https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html) and [poly1d](https://docs.scipy.org/doc/numpy/reference/generated/numpy.poly1d.html) operations to generate a linear function that matched the two input spaces (x and y). With this linear polynomial function, I could then enter the two known y values and generate two correct and matching x values. These (x, y) pairs were the beginning and end of the line segments that could then be drawn.

### 2. Identify potential shortcomings with your current pipeline

#### Line Segment Orientation

The shortcoming I immediately identify is that my algorithm assumes lane lines to be fairly consistent in terms of orientation. The values that I am using for the Hough transform require a large amount of consistency between detected edge points of each line in order to be detected as a line. Additionally, the line extrapolation operation makes a very broad assumption that lane lines are expected to have extreme slopes. This works most effectively when processing a mostly straight road with a lot of visibility. In the case of a curved road, the slopes of each detected Hough line constituent may be dampened as a part of the curve.

#### Line Segment Size / Gap Size

Another shortcoming may come from the assumption that lane lines usually have large contiguous edges to detect. Very small line segments with small gaps between them would be intuitively observed as a series by a human, but my algorithm would easily miss these lines. The source of this error is the assumption in my Hough transform that all lane lines will contain long contiguous segments.

### 3. Suggest possible improvements to your pipeline

#### Detection by "Ensemble"

I wanted to try running parallel pipelines on the imagery to attempt to gain the benefits of different options at each pipeline step. This "ensemble" approach could take the form of parallel processing on HSL color space as well as Grayscale. It would be interesting to see the results of applying different Hough transforms on the same imagery to target known classes of lane lines. This would be a good scaffolding for a machine learning system to learn the parameters necessary to detect labeled classes of lane line features.

#### Dynamic Cropping

A fixed region of interest is simpler to develop, because it uses static values to remove noise from the image being processed. This is effective given known values for things like the width of a lane or the size and visibility the hood of the car. In more varying circumstances, it would be an improvement to allow the algorithm to determine the region of interest based on detection or non-detection of lane line segments.

#### (Video-Only) Detected Line Momentum / Decay

There are potentially many frames of a given video where the tuned parameters do not detect any line segments whatsoever. In these cases, it is unlikely that the lane has actually disappeared. In order to handle frames with no detection effectively as well as smoothing out jitters in frame-by-frame analysis, we could implement some momentum for each lane line. This could come in the form of some rolling window of detected Hough lines from previous frames, or a more condensed representation of the linear functions for past lines could be used. In any case, we would likely want to include a fairly strong decay mechanism, because there are legitimate situations where the lanes do eventually end. This would help smooth out the information coming from the sensor doing this analysis.
