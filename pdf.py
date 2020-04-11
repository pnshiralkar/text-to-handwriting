# import img2pdf
#
# f = open('pages/page1.png', 'r')
#
# with open("handwritten.pdf", 'wb') as f:
#     f.write(img2pdf.convert(['pages/page1.jpg', 'pages/page2.jpg']))

from PIL import Image

im1 = Image.open("pages/page1.png")
im2 = Image.open("pages/page2.png")

im1.load()  # required for png.split()

background = Image.new("RGB", im1.size, (255, 255, 255))
background.paste(im1, mask=im1.split()[3])  # 3 is the alpha channel

background.save('pages/page1.jpg', 'JPEG', quality=100)
im1 = background
background = Image.new("RGB", im2.size, (255, 255, 255))
background.paste(im2, mask=im2.split()[3])  # 3 is the alpha channel

background.save('pages/page2.jpg', 'JPEG', quality=100)
im2 = background

im_list = [im2]

pdf1_filename = "handwritten.pdf"

im1.save(pdf1_filename, "PDF", resolution=100.0, save_all=True, append_images=im_list)
