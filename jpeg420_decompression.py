def jpg420recon(path):  
    image = jpegio.read(path)
    
    image.quant_tables.append(image.quant_tables[1])  # Cr 양자화 테이블 채널 추가(Cb와 동일, 실제 양자화 테이블은 2개) 
    coef_arrays = image.coef_arrays  # YCbCr 양자화된 DCT 계수
    quant_tables = image.quant_tables  # YCbCr 양자화 테이블


    h, w = coef_arrays[0].shape #Y 채널의 size 가져오기
    #print(h, w)
    image_q = np.array(quant_tables).astype(np.int16).reshape((-1, 1, 1, 8, 8)) # 8*8 쪼개기

    # coef_arrays[1] = coef_arrays[1].repeat(2, axis=0).repeat(2, axis=1)
    # coef_arrays[2] = coef_arrays[2].repeat(2, axis=0).repeat(2, axis=1)

    image_c_0 = np.array(coef_arrays[0]) #Y채널의 DCT 계수만 복사
    image_c_1 = np.array(coef_arrays[1]) #Cb채널의 DCT 계수만 복사
    image_c_2 = np.array(coef_arrays[2]) #Cr채널의 DCT 계수만 복사
    ## 각각 복사해서 진행하는 이유는 Y 채널과 Cb, Cr 채널의 크기가 맞지 않기 때문 
    ## YUV420에서 Y채널이 4*4라면 Cb, Cr 채널은 가로, 세로에 대해 각각 1/2 크기 이므로 2*2 size가 됨 
    ## 0 : Y , 1 : Cb , 2 : Cr
    
    image_c_0 = image_c_0.reshape((1, h // 8, 8, w // 8, 8), order='C').transpose(0, 1, 3, 2, 4)  # 8x8 block 단위로 채널 분리
    image_c_1 = image_c_1.reshape((1, (h//2) // 8, 8, (w//2) // 8, 8), order='C').transpose(0, 1, 3, 2, 4)
    image_c_2 = image_c_2.reshape((1, (h//2) // 8, 8, (w//2) // 8, 8), order='C').transpose(0, 1, 3, 2, 4)


    deq_0 = image_c_0 * image_q[0]  # 각각에 대해 양자화 복원: 양자화된 계수 * 양자화 테이블
    deq_1 = image_c_1 * image_q[1]
    deq_2 = image_c_2 * image_q[2]

    
    #print("deq2", deq_2.shape)
    image_ycbcr_0 = idct2d(deq_0).transpose(0, 1, 3, 2, 4).reshape(1, h, w)  # Inverse DCT2D 수행, 각각 한 채널씩 가져왔기 때문에 1, h, w로 진행 
    #YUV 444에서는 3으로 한번에 진행해도 무관 
    image_ycbcr_0 = image_ycbcr_0.transpose(1, 2, 0)  # 채널 변경: CxHwW -> HxWxC 

    image_ycbcr_1 = idct2d(deq_1).transpose(0, 1, 3, 2, 4).reshape(1, h//2, w//2)  ## h, w 값이 Y 채널에 대한 값이므로 2로 나누고 정수부분만 추출 
    image_ycbcr_1 = image_ycbcr_1.transpose(1, 2, 0)  
    image_ycbcr_2 = idct2d(deq_2).transpose(0, 1, 3, 2, 4).reshape(1, h//2, w//2) 
    image_ycbcr_2 = image_ycbcr_2.transpose(1, 2, 0)

    image_ycbcr_0 = cv2.resize(image_ycbcr_0, (h, w)) ##각 채널의 사이즈를 맞춰주기 위해 resize 사용, 기본적으로 INTER_LINEAR interpolation 수행
    image_ycbcr_1 = cv2.resize(image_ycbcr_1, (h, w)) 
    image_ycbcr_2 = cv2.resize(image_ycbcr_2, (h, w))

    image_ycbcr = np.stack((image_ycbcr_0, image_ycbcr_1, image_ycbcr_2), axis=0)  ## 분리된 Y, Cb, Cr 채널 합치기 
    image_ycbcr = image_ycbcr.transpose(1, 2, 0) ## 원하는 h, w, c 순으로 변경하기 위함

    #print(image_ycbcr_0.shape, image_ycbcr_1.shape)
    image_ycbcr[:, :, 0] += 128  # 밝기 Level Shift 복원 (128 더하기)
    
    image_rgb = np.clip(ycbcr2rgb(image_ycbcr), 0, 255)  # RGB 픽셀값 범위로 값 절삭
    image_rgb = image_rgb.astype(np.uint8)  # UINT8로 타입 변환 (값 범위 0~255)

    return image_rgb
  
img_color_recon420 = jpg420recon()  # jpegio로 이미지 불러와서 RGB로 복원하는 함수 실행, ()안에 image path 추가
                                    #코드 원리를 위한 image >> test_image 폴더에 elon-face.jpg 업로드 해둠
plt.imshow(img_color_recon420)  # 이미지 화면 출력
