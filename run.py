import cv2

from app import create_app
from app.services.hologram_service import grid_analysis, detect_and_process_faces

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=6060)
    # inputimg = 'C:/Users/Yun/PycharmProjects/PythonProject/ProjectPhoto.jpg'
    #
    # # 做臉部偵測與區塊切割
    # img, regions = detect_and_process_faces(inputimg)
    #
    # if img is not None:
    #
    #     # 各區塊個別分析
    #     for roi_img, label in regions:
    #         analysis = grid_analysis(roi_img, name=label)
    #         cv2.imshow(f"{label} - 格狀分析", analysis['grid'])
    #
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()