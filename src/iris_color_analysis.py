import cv2
import matplotlib
matplotlib.use('TkAgg')   # set backend BEFORE pyplot
import matplotlib.pyplot as plt
import numpy as np
import math
import os


# Folder containing timestamp images (WSL path for D:)
folder = "/mnt/d/scienceday"


def add_ticks(image):

    h, w = image.shape[:2]

    step_x = max(1, w // 6)
    step_y = max(1, h // 8)

    plt.xticks(np.arange(0, w, step_x))
    plt.yticks(np.arange(0, h, step_y))


# Get all jpg files
image_files = sorted(
    [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]
)

if not image_files:
    print("No JPG images found in folder.")
    exit()

# Select latest timestamp image
latest_file = image_files[-1]
image_path = os.path.join(folder, latest_file)

for file in [latest_file]:

    print("\nTesting:", image_path)

    img = cv2.imread(image_path)

    if img is None:
        print("Skipped (not an image)")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eye_gray = cv2.equalizeHist(gray)
    eye_gray = cv2.GaussianBlur(eye_gray,(5,5),0)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    face_output = img.copy()
    face_eye_output = img.copy()
    eye_count = 0
    cropped_eyes = []

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(face_output,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(face_output,"Face",(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

            cv2.rectangle(face_eye_output,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(face_eye_output,"Face",(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

            roi_gray = eye_gray[y:y+h//2,x:x+w]
            roi_color = face_eye_output[y:y+h//2,x:x+w]

            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.05,
                minNeighbors=10,
                minSize=(20,20)
            )

            eye_count += len(eyes)

            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),
                              (ex+ew,ey+eh),(255,0,0),2)
                cv2.putText(roi_color,"Eye",(ex,ey-5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,(255,0,0),2)

                eye_crop = img[y+ey:y+ey+eh, x+ex:x+ex+ew]
                eye_resized = cv2.resize(eye_crop,(100,100))
                cropped_eyes.append(eye_resized)

    else:
        print("No face detected — scanning entire image for eyes")
        eyes = eye_cascade.detectMultiScale(
            eye_gray,
            scaleFactor=1.05,
            minNeighbors=10,
            minSize=(20,20)
        )
        eye_count += len(eyes)

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(face_eye_output,
                          (ex,ey),
                          (ex+ew,ey+eh),
                          (255,0,0),2)
            cv2.putText(face_eye_output,
                        "Eye",
                        (ex,ey-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255,0,0),
                        2)
            eye_crop = img[ey:ey+eh, ex:ex+ew]
            eye_resized=cv2.resize(eye_crop,(100,100))
            cropped_eyes.append(eye_resized)

    print(f"Number of faces detected: {len(faces)}")
    print(f"Number of eyes detected: {eye_count}")

    # === Eye grid panel ===
    if cropped_eyes:
        n=len(cropped_eyes)
        cols=math.ceil(math.sqrt(n))
        rows=math.ceil(n/cols)
        blank=np.zeros((100,100,3),dtype=np.uint8)
        while len(cropped_eyes)<rows*cols:
            cropped_eyes.append(blank)
        grid_rows=[]
        for r in range(rows):
            row=cropped_eyes[r*cols:(r+1)*cols]
            grid_rows.append(cv2.hconcat(row))
        eyes_panel=cv2.vconcat(grid_rows)
    else:
        eyes_panel=np.zeros((200,200,3),dtype=np.uint8)

    # === Refined eyes ===
    refined_eyes=[]
    for eye in cropped_eyes:
        h,w,_=eye.shape
        top=int(0.30*h)
        bottom=int(0.75*h)
        left=int(0.15*w)
        right=int(0.85*w)
        tight_eye=eye[top:bottom,left:right]
        tight_eye=cv2.resize(tight_eye,(100,100))
        refined_eyes.append(tight_eye)

    if refined_eyes:
        n2=len(refined_eyes)
        cols2=math.ceil(math.sqrt(n2))
        rows2=math.ceil(n2/cols2)
        blank2=np.zeros((100,100,3),dtype=np.uint8)
        while len(refined_eyes)<rows2*cols2:
            refined_eyes.append(blank2)
        grid_rows2=[]
        for r in range(rows2):
            row=refined_eyes[r*cols2:(r+1)*cols2]
            grid_rows2.append(cv2.hconcat(row))
        refined_panel=cv2.vconcat(grid_rows2)
    else:
        refined_panel=np.zeros((200,200,3),dtype=np.uint8)

    # === Daugman iris detection ===
    iris_outputs=[]
    iris_circles=[]
    for eye in refined_eyes:
        gray_eye=cv2.cvtColor(eye,cv2.COLOR_BGR2GRAY)
        gray_eye=cv2.GaussianBlur(gray_eye,(5,5),1)
        h,w=gray_eye.shape
        gx=cv2.Sobel(gray_eye,cv2.CV_64F,1,0,ksize=3)
        gy=cv2.Sobel(gray_eye,cv2.CV_64F,0,1,ksize=3)
        gradient=np.sqrt(gx**2+gy**2)
        best_score=0
        best_circle=None
        r_min=int(0.20*w)
        r_max=int(0.45*w)
        for cx in range(int(0.3*w),int(0.7*w),2):
            for cy in range(int(0.3*h),int(0.7*h),2):
                for r in range(r_min,r_max,2):
                    score=0
                    samples=40
                    for t in range(samples):
                        theta=2*np.pi*t/samples
                        x=int(cx+r*np.cos(theta))
                        y=int(cy+r*np.sin(theta))
                        if 0<=x<w and 0<=y<h:
                            score+=gradient[y,x]
                    if score>best_score:
                        best_score=score
                        best_circle=(cx,cy,r)
        output=eye.copy()
        if best_circle:
            cx,cy,r=best_circle
            cv2.circle(output,(cx,cy),r,(0,255,0),2)
            cv2.circle(output,(cx,cy),2,(0,0,255),2)
            iris_circles.append((eye,cx,cy,r))
        iris_outputs.append(output)

    if iris_outputs:
        n3=len(iris_outputs)
        cols3=math.ceil(math.sqrt(n3))
        rows3=math.ceil(n3/cols3)
        blank3=np.zeros((100,100,3),dtype=np.uint8)
        while len(iris_outputs)<rows3*cols3:
            iris_outputs.append(blank3)
        grid_rows3=[]
        for r in range(rows3):
            row=iris_outputs[r*cols3:(r+1)*cols3]
            grid_rows3.append(cv2.hconcat(row))
        iris_panel=cv2.vconcat(grid_rows3)
    else:
        iris_panel=np.zeros((200,200,3),dtype=np.uint8)

    # === Iris pixels collection ===
    iris_images=[]
    iris_nosclera_images=[]
    iris_pixel_array=[]
    iris_gray_array=[]
    iris_rgb_nosclera=[]
    iris_hsv_nosclera=[]

    for (eye,cx,cy,r) in iris_circles:
        h,w,_=eye.shape
        iris_only=np.zeros_like(eye)
        iris_nosclera=np.zeros_like(eye)
        hsv=cv2.cvtColor(eye,cv2.COLOR_BGR2HSV)
        for row in range(h):
            for col in range(w):
                if (col-cx)**2+(row-cy)**2<=r*r:
                    B,G,R=eye[row,col]
                    iris_only[row,col]=eye[row,col]
                    iris_pixel_array.append([col,row,int(R),int(G),int(B)])
                    gray_val=0.114*B+0.587*G+0.299*R
                    iris_gray_array.append([col,row,gray_val])
                    H,S,V=hsv[row,col]
                    sclera=False
                    if gray_val>170 and S<40:
                        sclera=True
                    if not sclera:
                        iris_nosclera[row,col]=eye[row,col]
                        iris_rgb_nosclera.append([col,row,int(R),int(G),int(B)])
                        iris_hsv_nosclera.append([col,row,int(H),int(S),int(V)])
        iris_images.append(iris_only)
        iris_nosclera_images.append(iris_nosclera)

    if iris_images:
        iris_panel2=cv2.hconcat(iris_images[:2])
    else:
        iris_panel2=np.zeros((100,200,3),dtype=np.uint8)
    if iris_nosclera_images:
        iris_panel3=cv2.hconcat(iris_nosclera_images[:2])
    else:
        iris_panel3=np.zeros((100,200,3),dtype=np.uint8)

    print("Total iris pixels:",len(iris_pixel_array))
    print("Iris pixels without sclera:",len(iris_rgb_nosclera))

    # === Visualization panels ===
    fig=plt.figure(figsize=(12,9))
    panels=[
        (img,"Input Image"),
        (face_output,"Detected Faces"),
        (face_eye_output,"Faces & Eyes"),
        (eyes_panel,"Eye Grid"),
        (refined_panel,"Refined Eyes"),
        (iris_panel,"Panel 6: Daugman Iris"),
        (iris_panel2,"Panel 7: Iris Only"),
        (iris_panel3,"Panel 8: Iris Without Sclera")
    ]
    axes=[]
    for i,(im,title) in enumerate(panels):
        ax=plt.subplot(3,3,i+1)
        rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(title)
        add_ticks(im)
        axes.append((ax,rgb))

    crosshair_h=[]
    crosshair_v=[]
    for ax,_ in axes:
        crosshair_h.append(ax.axhline(color='white'))
        crosshair_v.append(ax.axvline(color='white'))

    def mouse_move(event):
        if event.inaxes is None:
            return
        x=int(event.xdata)
        y=int(event.ydata)
        ax_hover = event.inaxes
        idx = [a for a,_ in axes].index(ax_hover)
        imgdata = axes[idx][1]
        crosshair_h[idx].set_ydata([y,y])
        crosshair_v[idx].set_xdata([x,x])
        if 0<=x<imgdata.shape[1] and 0<=y<imgdata.shape[0]:
            R,G,B=imgdata[y,x]
            fig.suptitle(f"x={x}  y={y}   R={R} G={G} B={B}",fontsize=12)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event',mouse_move)
    plt.tight_layout(rect=[0,0,1,0.94])
    plt.show()
    
    # === Figure 2: RGB + HSV Histograms Combined ===

    if len(iris_rgb_nosclera) > 0 and len(iris_hsv_nosclera) > 0:

        # -------- RGB PART --------
        rgb_array = np.array(iris_rgb_nosclera)
        R_vals = rgb_array[:,2]
        G_vals = rgb_array[:,3]
        B_vals = rgb_array[:,4]

        hist_R, _ = np.histogram(R_vals, bins=256, range=(0,256))
        hist_G, _ = np.histogram(G_vals, bins=256, range=(0,256))
        hist_B, _ = np.histogram(B_vals, bins=256, range=(0,256))

        x_rgb = np.arange(256)

        # -------- HSV PART --------
        hsv_array = np.array(iris_hsv_nosclera)

        H_vals = hsv_array[:,2] * 2
        S_vals = hsv_array[:,3]
        V_vals = hsv_array[:,4]

        hist_H, _ = np.histogram(H_vals, bins=361, range=(0,361))
        hist_S, _ = np.histogram(S_vals, bins=256, range=(0,256))
        hist_V, _ = np.histogram(V_vals, bins=256, range=(0,256))

        x_H = np.arange(361)
        x_S = np.arange(256)
        x_V = np.arange(256)

        # -------- Combined Figure --------
        fig2 = plt.figure(figsize=(16,8))

        # ===== Row 1 : RGB =====
        ax1 = plt.subplot(2,4,1)
        colors_R = [(i/255, 0, 0) for i in x_rgb]
        ax1.bar(x_rgb, hist_R, color=colors_R, width=1.0)
        ax1.set_title("Red Channel Histogram")
        ax1.set_xlabel("Red (0-255)")
        ax1.set_ylabel("Frequency")
        ax1.set_xlim(0,255)
        ax1.grid(True)

        ax2 = plt.subplot(2,4,2)
        colors_G = [(0, i/255, 0) for i in x_rgb]
        ax2.bar(x_rgb, hist_G, color=colors_G, width=1.0)
        ax2.set_title("Green Channel Histogram")
        ax2.set_xlabel("Green (0-255)")
        ax2.set_ylabel("Frequency")
        ax2.set_xlim(0,255)
        ax2.grid(True)

        ax3 = plt.subplot(2,4,3)
        colors_B = [(0, 0, i/255) for i in x_rgb]
        ax3.bar(x_rgb, hist_B, color=colors_B, width=1.0)
        ax3.set_title("Blue Channel Histogram")
        ax3.set_xlabel("Blue (0-255)")
        ax3.set_ylabel("Frequency")
        ax3.set_xlim(0,255)
        ax3.grid(True)

        ax4 = plt.subplot(2,4,4)
        ax4.plot(x_rgb, hist_R, color='red', label='Red')
        ax4.plot(x_rgb, hist_G, color='green', label='Green')
        ax4.plot(x_rgb, hist_B, color='blue', label='Blue')
        ax4.set_title("Combined RGB Histogram")
        ax4.set_xlabel("Intensity (0-255)")
        ax4.set_ylabel("Frequency")
        ax4.set_xlim(0,255)
        ax4.legend()
        ax4.grid(True)

        # ===== Row 2 : HSV =====
        ax5 = plt.subplot(2,4,5)
        colors_H = [plt.cm.hsv(i/360) for i in x_H]
        ax5.bar(x_H, hist_H, color=colors_H, width=1.0)
        ax5.set_title("Hue Histogram (0-360)")
        ax5.set_xlabel("Hue (Degrees)")
        ax5.set_ylabel("Frequency")
        ax5.set_xlim(0,360)
        ax5.grid(True)

        ax6 = plt.subplot(2,4,6)
        colors_S = [(i/255, i/255, i/255) for i in x_S]
        ax6.bar(x_S, hist_S, color=colors_S, width=1.0)
        ax6.set_title("Saturation Histogram")
        ax6.set_xlabel("S (0-255)")
        ax6.set_ylabel("Frequency")
        ax6.set_xlim(0,255)
        ax6.grid(True)

        ax7 = plt.subplot(2,4,7)
        colors_V = [(i/255, i/255, i/255) for i in x_V]
        ax7.bar(x_V, hist_V, color=colors_V, width=1.0)
        ax7.set_title("Value Histogram")
        ax7.set_xlabel("V (0-255)")
        ax7.set_ylabel("Frequency")
        ax7.set_xlim(0,255)
        ax7.grid(True)

        ax8 = plt.subplot(2,4,8)
        ax8.plot(x_H, hist_H, color='magenta', label='Hue')
        ax8.plot(x_S, hist_S, color='black', label='Saturation')
        ax8.plot(x_V, hist_V, color='gray', label='Value')
        ax8.set_title("Combined HSV Histogram")
        ax8.set_xlabel("Intensity / Hue")
        ax8.set_ylabel("Frequency")
        ax8.set_xlim(0,360)
        ax8.legend()
        ax8.grid(True)

        plt.tight_layout()
        plt.show()

    else:
        print("RGB or HSV iris pixels unavailable.")

    # === Figure 3: Clean RGB + HSV Spaces + KMeans Spheres ===

    if len(iris_rgb_nosclera) > 0 and len(iris_hsv_nosclera) > 0:

        import matplotlib
        matplotlib.use('TkAgg')

        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from sklearn.cluster import KMeans
        import colorsys

        fig3 = plt.figure(figsize=(14,12))

        # ===================== RGB VOXELS =====================

        rgb_array = np.array(iris_rgb_nosclera)

        # OpenCV stores BGR → convert properly
        R_vals = rgb_array[:,2]
        G_vals = rgb_array[:,3]
        B_vals = rgb_array[:,4]

        unique_rgb = np.unique(
            np.stack((R_vals, G_vals, B_vals), axis=1),
            axis=0
        )

        ax1 = fig3.add_subplot(221, projection='3d')

        ax1.scatter(
            unique_rgb[:,0],
            unique_rgb[:,1],
            unique_rgb[:,2],
            c=unique_rgb/255.0,
            s=8,
            depthshade=False
        )

        ax1.set_title("RGB Space (Real Iris Voxels)")
        ax1.set_xlabel("Red")
        ax1.set_ylabel("Green")
        ax1.set_zlabel("Blue")

        ax1.set_xlim(0,255)
        ax1.set_ylim(0,255)
        ax1.set_zlim(0,255)
        ax1.grid(True)

        # ===================== HSV VOXELS =====================

        hsv_array = np.array(iris_hsv_nosclera)

        H_vals = hsv_array[:,2]      # OpenCV 0–179
        S_vals = hsv_array[:,3]
        V_vals = hsv_array[:,4]

        H_deg = H_vals * 2           # Convert to 0–360

        # Convert HSV → RGB for human perception
        rgb_colors = []
        for h, s, v in zip(H_deg, S_vals, V_vals):
            r, g, b = colorsys.hsv_to_rgb(h/360.0, s/255.0, v/255.0)
            rgb_colors.append((r,g,b))

        rgb_colors = np.array(rgb_colors)

        ax2 = fig3.add_subplot(222, projection='3d')

        ax2.scatter(
            H_deg,
            S_vals,
            V_vals,
            c=rgb_colors,
            s=6,
            marker='s',
            depthshade=False
        )

        ax2.set_title("HSV Space (Hue 0–360)")
        ax2.set_xlabel("Hue (0–360)")
        ax2.set_ylabel("Saturation")
        ax2.set_zlabel("Value")

        ax2.set_xlim(0,360)
        ax2.set_ylim(0,255)
        ax2.set_zlim(0,255)
        ax2.grid(True)

        # ===================== RGB KMEANS =====================

        kmeans_rgb = KMeans(n_clusters=3, random_state=0, n_init=10)
        labels_rgb = kmeans_rgb.fit_predict(unique_rgb)
        centers_rgb = kmeans_rgb.cluster_centers_

        counts_rgb = np.bincount(labels_rgb)
        total_rgb = np.sum(counts_rgb)
        percent_rgb = (counts_rgb / total_rgb) * 100
        order_rgb = np.argsort(counts_rgb)   # least → most
        sizes = [400, 800, 1400]

        ax3 = fig3.add_subplot(223, projection='3d')

        for rank, idx in enumerate(order_rgb):

            r_c, g_c, b_c = centers_rgb[idx]

            pct = percent_rgb[idx]

            ax3.scatter(
                r_c,
                g_c,
                b_c,
                s=sizes[rank],
                c=[[r_c/255.0, g_c/255.0, b_c/255.0]],
                edgecolors='black',
                linewidths=2,
                depthshade=False
            )

            ax3.text(
                r_c,
                g_c,
                b_c,
                f"K{rank+1} {pct:.1f}%",
                fontsize=11,
                color='black'
            )

        ax3.set_title("RGB K-Means Cluster Centers")
        ax3.set_xlabel("Red")
        ax3.set_ylabel("Green")
        ax3.set_zlabel("Blue")

        ax3.set_xlim(0,255)
        ax3.set_ylim(0,255)
        ax3.set_zlim(0,255)
        ax3.grid(True)

        # ===================== HSV KMEANS =====================

        hsv_for_kmeans = np.column_stack([H_deg, S_vals, V_vals])

        kmeans_hsv = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels_hsv = kmeans_hsv.fit_predict(hsv_for_kmeans)
        centers_hsv = kmeans_hsv.cluster_centers_

        counts_hsv = np.bincount(labels_hsv)
        total_hsv = np.sum(counts_hsv)
        percent_hsv = (counts_hsv / total_hsv) * 100
        order_hsv = np.argsort(counts_hsv)

        ax4 = fig3.add_subplot(224, projection='3d')

        for rank, idx in enumerate(order_hsv):

            h_c, s_c, v_c = centers_hsv[idx]

            pct = percent_hsv[idx]

            r, g, b = colorsys.hsv_to_rgb(h_c/360.0, s_c/255.0, v_c/255.0)

            ax4.scatter(
                h_c,
                s_c,
                v_c,
                s=sizes[rank],
                c=[[r,g,b]],
                edgecolors='black',
                linewidths=2,
                depthshade=False
            )

            ax4.text(
                h_c,
                s_c,
                v_c,
                f"K{rank+1} {pct:.1f}%",
                fontsize=11,
                color='black'
            )

        ax4.set_title("HSV K-Means Cluster Centers")
        ax4.set_xlabel("Hue (0–360)")
        ax4.set_ylabel("Saturation")
        ax4.set_zlabel("Value")

        ax4.set_xlim(0,360)
        ax4.set_ylim(0,255)
        ax4.set_zlim(0,255)
        ax4.grid(True)

        plt.tight_layout(pad=3.0)
        plt.show()

    else:
        print("RGB or HSV iris pixels unavailable.")

    # ===========================
    # === Figure 4: Eye Color Table
    # ===========================

    if len(iris_hsv_nosclera) > 0:

        # Ensure K1 = dominant cluster
        order_hsv = np.argsort(counts_hsv)[::-1]

        total_pixels = np.sum(counts_hsv)
        percent_hsv = (counts_hsv / total_pixels) * 100


        # ======================================
        # Eye Color Classification Function
        # ======================================

        def classify_eye_color(H,S,V):

            if S < 25:
                if V > 170:
                    return "Albinism / Very Light"
                else:
                    return "Grey"

            if V < 60:
                return "Dark Brown"

            if H < 25 or H > 340:

                if V < 110:
                    return "Dark Brown"

                elif V < 160:
                    return "Brown"

                else:
                    return "Light Brown"

            if 25 <= H <= 45:
                return "Amber"

            if 45 < H <= 140:
                return "Green"

            if 140 < H <= 260:
                return "Blue"

            return "Grey"


        # Build Cluster Table

        table_rows = []

        for i in range(3):

            idx = order_hsv[i]

            h_c, s_c, v_c = centers_hsv[idx]

            pct = percent_hsv[idx]

            color_name = classify_eye_color(h_c,s_c,v_c)

            table_rows.append([
                f"K{i+1}",
                f"{pct:.1f} %",
                f"{int(h_c)}",
                f"{int(s_c)}",
                f"{int(v_c)}",
                color_name
            ])


        # Statistical Table (RGB + HSV)

        from scipy.stats import kurtosis, skew
        from scipy.signal import find_peaks

        rgb_array = np.array(iris_rgb_nosclera)
        hsv_array = np.array(iris_hsv_nosclera)

        R = rgb_array[:,2]
        G = rgb_array[:,3]
        B = rgb_array[:,4]

        H = hsv_array[:,2]*2
        S = hsv_array[:,3]
        V = hsv_array[:,4]


        def entropy_calc(data):

            hist,_ = np.histogram(data,bins=256)

            p = hist/np.sum(hist)

            p = p[p>0]

            return -np.sum(p*np.log2(p))


        def peak_stats(data):

            hist,_ = np.histogram(data,bins=256)

            peaks,_ = find_peaks(hist,height=max(hist)*0.1)

            n_peaks = len(peaks)

            if n_peaks>0:
                peakwidth = np.std(peaks)
            else:
                peakwidth = 0

            return n_peaks,peakwidth


        def stats_row(data):

            n_peaks,peakwidth = peak_stats(data)

            return [

                f"{np.mean(data):.1f}",
                f"{np.median(data):.1f}",
                f"{np.bincount(data.astype(int)).argmax()}",
                f"{np.std(data):.1f}",
                f"{np.median(np.abs(data-np.median(data))):.1f}",
                f"{kurtosis(data):.2f}",
                f"{skew(data):.2f}",
                f"{np.percentile(data,75)-np.percentile(data,25):.1f}",
                f"{entropy_calc(data):.2f}",
                f"{peakwidth:.1f}",
                f"{n_peaks}"

            ]


        stats_table = [

            ["R"] + stats_row(R),
            ["G"] + stats_row(G),
            ["B"] + stats_row(B),
            ["H"] + stats_row(H),
            ["S"] + stats_row(S),
            ["V"] + stats_row(V)

        ]


        # Create Figure 4

        fig4 = plt.figure(figsize=(14,12))

        ax = plt.subplot(111)

        ax.axis('off')


        # -------- Cluster Table --------

        table = ax.table(

            cellText = table_rows,

            colLabels=[

                "Cluster",
                "Percent",
                "Hue",
                "Sat",
                "Val",
                "Eye Color"

            ],

            bbox=[0.05,0.65,0.9,0.25]

        )

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2,1.8)


        # -------- Statistics Table --------

        table2 = ax.table(

            cellText = stats_table,

            colLabels=[

                "Channel",
                "Mean",
                "Median",
                "Mode",
                "Std",
                "MAD",
                "Kurt",
                "Skew",
                "IQR",
                "Entropy",
                "PeakW",
                "#Peaks"

            ],

            bbox=[0.05,0.05,0.9,0.50]

        )

        table2.auto_set_font_size(False)
        table2.set_fontsize(10)
        table2.scale(1.0,1.3)


        # -------- Title --------

        dominant_idx = order_hsv[0]

        domH,domS,domV = centers_hsv[dominant_idx]

        dominant_color = classify_eye_color(domH,domS,domV)

        plt.title(

            f"Figure 4 — Iris Color Classification\nDominant Eye Color: {dominant_color}",

            fontsize=16

        )

        plt.show()

    else:

        print("No iris HSV data for Figure 4")