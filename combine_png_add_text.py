from PIL import Image, ImageDraw, ImageFont
import os
import glob
import json
import matplotlib.pyplot as plt
import io


def generate_timelapse_2D_evaluation_error_map():
    png_sorce_path = r'F:\CMap_paper\Code\Evaluation\Results\2DErrorMap\200109plc1p1'
    dst_png_path = r'F:\CMap_paper\Code\Evaluation\Results\2DErrorMap\SnapTextReverseAdded\200109plc1p1'
    png_embryo_list1 = glob.glob(os.path.join(png_sorce_path, '*.png'))
    raw_seg_gui_path = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\MembraneProjectData\GUIData\WebData_CMap_cell_label_v3'

    # print(tif_embryo_list)
    # cell_number_this = 21

    mask_width_start = 50
    mask_height_start = 30
    mask_width_stop = 560
    mask_height_stop = 366
    mask = (mask_width_start, mask_height_start, mask_width_stop, mask_height_stop)

    text_height = 50


    for idx, png_snaps1 in enumerate(png_embryo_list1):
        embryo_name, embryo_tp = os.path.basename(png_snaps1).split('_')[:2]

        png1 = Image.open(png_snaps1).crop(mask)

        # Create a new image with the same mode and size as the first image
        # text_height = 150
        # top_text_image=upper_right_png.crop((0,0,))
        combined_image = Image.new(png1.mode, (png1.width * 1, text_height + png1.height * 1),
                                   color=(20, 0, 78))

        # Paste the images into the new image
        combined_image.paste(png1, (0, 0 + text_height))
        # width, height = 1920, 1150

        tp_cell_file_path = os.path.join(raw_seg_gui_path, embryo_name, 'TPCell',
                                         embryo_name + '_' + embryo_tp + '_cells.txt')
        # Open the file for reading
        with open(tp_cell_file_path, 'r') as file:
            # Read the contents of the file into a string
            contents = file.read()
            # Split the string into a list using the comma as the delimiter
            my_list = contents.split(',')

        # time_this_tp = "{:.2f}".format((idx + 1) * 1.43)
        cell_number_this_tp = str(len(my_list))
        time_this_tp = "{:.2f}".format((int(embryo_tp) -1) * 1.43)
        draw = ImageDraw.Draw(combined_image)
        font = ImageFont.truetype('arial.ttf', size=25)
        text1 = 'Time: ' + time_this_tp + ' min' + ' ; ' + 'Total Cell Number: ' + cell_number_this_tp
        # text2 =
        textwidth, textheight = draw.textsize(text1, font)
        width, height = combined_image.size
        x = (width - textwidth) // 2
        y = textheight*0.8
        draw.text((x, y), text1, fill="white", font=font)

        combined_image.save(os.path.join(dst_png_path, str(idx).zfill(3) + '.png'))

def generate_timelapse_3D_evaluation_error_map():
    # 200113plc1p2 for tuning_cell_text_height -20,textheti 100,,, 200109plc1p1 for tuning_cell_text_height +20 textheight50
    embryo_name_gt_3d='200113plc1p2'
    png_sorce_path = r'F:\CMap_paper\Code\Evaluation\Results\3DErrorMap'
    folder_list=glob.glob(os.path.join(png_sorce_path,'{}*'.format(embryo_name_gt_3d)))
    file_list=[]
    for item_path in folder_list:
        file_list.append(glob.glob(os.path.join(item_path,'*.png')))
    print(file_list)

    # print(tif_embryo_list)
    # cell_number_this = 21

    mask_width_start = 50
    mask_height_start = 30
    mask_width_stop = 560
    mask_height_stop = 366
    mask = (mask_width_start, mask_height_start, mask_width_stop, mask_height_stop)

    color_bar_height=100
    text_height = 100
    tuning_cell_text_height=-20

    for idx in range(256):
        png0 = Image.open(file_list[0][idx]).crop(mask)
        png1 = Image.open(file_list[1][idx]).crop(mask)
        png2 = Image.open(file_list[2][idx]).crop(mask)
        png3 = Image.open(file_list[3][idx]).crop(mask)
        png4 = Image.open(file_list[4][idx]).crop(mask)
        png5 = Image.open(file_list[5][idx]).crop(mask)


        # Create a new image with the same mode and size as the first image
        # text_height = 150
        # top_text_image=upper_right_png.crop((0,0,))
        combined_image = Image.new(png0.mode, (png0.width * 3, text_height + png0.height * 2+color_bar_height),
                                   color=(0, 0, 0))

        # Paste the images into the new image
        combined_image.paste(png0, (0, 0 + text_height))
        combined_image.paste(png1, (png0.width * 1, 0 + text_height))
        combined_image.paste(png2, (png0.width * 2, 0 + text_height))
        combined_image.paste(png3, (0, png0.height + text_height))
        combined_image.paste(png4, (png0.width * 1, png0.height + text_height))
        combined_image.paste(png5, (png0.width * 2, png0.height + text_height))
        # width, height = 1920, 1150

        draw = ImageDraw.Draw(combined_image)
        width, height = combined_image.size

        font = ImageFont.truetype('arial.ttf', size=40)
        # if idx==0:
        text1 = str(idx+1)+r'$ slice along   x   axis$'
        # # text2 =
        textwidth, textheight = draw.textsize(text1, font)
        x = (width - textwidth) // 2
        y = textheight*0.2
        # draw.text((x, y), text1, fill="white", font=font)


        # plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'arial'
        plt.rcParams['font.size'] = 30
        fig = plt.figure()
        index_string = 'Slice '+str(idx + 1)

        fig.text(0, 0.1, index_string+r' along ${x}$ axis', color='white')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='black')
        buf.seek(0)
        equation_image = Image.open(buf)

        # Now, open the original image and paste the equation image onto it
        combined_image.paste(equation_image, (x+50, -380))

        # write cell stage text
        font = ImageFont.truetype('arial.ttf', size=30)
        text_cell_stage = '~100-cell'
        textwidth, textheight = draw.textsize(text_cell_stage, font)
        x_interval=width//6
        draw.text((x_interval*1-textwidth//2, text_height+tuning_cell_text_height), text_cell_stage, fill="white", font=font)

        text_cell_stage = '~200-cell'
        draw.text((x_interval*3-textwidth//2, text_height+tuning_cell_text_height), text_cell_stage, fill="white", font=font)

        text_cell_stage = '~300-cell'
        draw.text((x_interval * 5-textwidth//2, text_height+tuning_cell_text_height), text_cell_stage, fill="white", font=font)

        text_cell_stage = '~400-cell'
        draw.text((x_interval * 1-textwidth//2, text_height+png0.height+tuning_cell_text_height), text_cell_stage, fill="white", font=font)
        text_cell_stage = '~500-cell'
        draw.text((x_interval * 3-textwidth//2, text_height + png0.height+tuning_cell_text_height), text_cell_stage, fill="white", font=font)
        text_cell_stage = '~550-cell'
        draw.text((x_interval * 5-textwidth//2, text_height + png0.height+tuning_cell_text_height), text_cell_stage, fill="white", font=font)


        # draw color map on the right
        # Define the colors for the seismic colorbar
        colors = [(0,0,165,178),(0,0,203,178),(0,0,243,178), (37,37,255,178),(136,136,255,178),(193,193,255,178),
                  (255,255,255,178),
                  (255,193,193,178),(255,136,136,178),(255,36,36,178),(243,0,0,178),(203,0,0,178),(165,0,0,178)]

        # Draw the colorbar

        for i in range(width//4,width//4*3):
            x0 = i
            x1 = i+1
            y0 = text_height + png0.height * 2+color_bar_height//4
            y1 = text_height + png0.height * 2+color_bar_height//2
            color_index =int((i-width//4) / (width//2) * len(colors))
            draw.rectangle([x0, y0, x1, y1], fill=colors[color_index])

        text1='Small difference'
        textwidth, textheight = draw.textsize(text1, font)
        draw.text((width//4, y1), text1, fill="white", font=font)

        text2='Big difference'
        textwidth, textheight = draw.textsize(text2, font)

        draw.text((width//4*3-textwidth, y1), text2, fill="white", font=font)



        combined_image.save(os.path.join(r'F:\CMap_paper\Figures\ErrorMap\{}'.format(embryo_name_gt_3d), str(idx).zfill(3) + '_x.png'))

if __name__=='__main__':
    generate_timelapse_3D_evaluation_error_map()