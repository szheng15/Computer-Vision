from PIL import Image
from pylab import *

im = Image.open('empire.jpg')

im.show()



empire_cropped = im.crop((100,100,400,400))

empire_cropped = empire_cropped.resize((128,128))

empire_cropped.show()




empire_cropped_rotated = empire_cropped.rotate(45)

empire_cropped_rotated.show()




empire_cropped.save('empire_cropped.jpg')

empire_cropped_rotated.save('empire_cropped_rotated.jpg')




imarray = array(Image.open('empire.jpg'))

imshow(imarray)

x = [100, 100, 400, 400]
y = [200, 500, 200, 500]

plot(x,y,'r*')

plot(x[:2],y[:2])

title('Plotting: "empire.jpg"')

show()


empire_gray = Image.open('empire.jpg').convert('L')

empire_gray.show()

empire_gray.save('empire_gray.jpg')


#Question 1:


