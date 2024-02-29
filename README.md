# Ball Tracking

Working on `ballTracking.py`.

## Notes
1. GrayFrame can detect some of the balls that my mask cannot. However it is quite useles, currently. Maybe if i get a range of where the ball should be then grayframe could only search a smaller area. GrayFrame could adopt findApproximateCircle aswell then.
2. Mask filtering is a little off currently. It needs a wider range of values.

## Next Steps

1. ~~Start video when the rally has started!~~
2. Try to make better ball detection with current filters.
    ~~2.1. Add higher contrast~~
    ~~2.2. Add sharpening kernel~~  

3. Add HSV filter to look for ball color for additional detection.
    - If circle detected, then ball should be green.

4. HoughCircle expects hollow circle -> hollow my masked ball by finding the perimeters of it!

5. My HoughLines is not working correctly... Or the draw function...