import cv2

def main():
    f = open('flir_local.csv')
    for l in f.readlines():
        path,x1, y1, x2, y2,classid = l.split(',')
        img = cv2.imread(path)
        if x1 == '':
            continue
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        rect = cv2.rectangle(img, (x1, y1), (x2, y2), (36,255,12), 1)
        image = cv2.putText(rect, classid,(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.imshow('prova', image)
        cv2.waitKey()

if __name__ == '__main__':
    main()