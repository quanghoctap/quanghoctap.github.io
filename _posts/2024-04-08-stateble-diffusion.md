---
title: 'Paper Explained 6: High Resolution Image Synthesis with Latent Diffusion Models'
date: 2024-04-08
categories: [Data Science, Deep Learning]
tags: [advanced, paper explained]     # TAG names should always be lowercase
toc: true
math: true
publish: true
---

Stable Diffusion là một mô hình trí tuệ nhân tạo (AI) tạo sinh được phát triển bởi Stability AI. Nó cho phép bạn tạo ra hình ảnh từ mô tả bằng văn bản. Stable Diffusion hoạt động dựa trên mô hình khuếch tán, bắt đầu với một bức tranh nhiễu và dần dần "khuếch tán" nhiễu đó để tạo ra hình ảnh mong muốn. Quá trình này được điều khiển bởi mô tả bằng văn bản mà bạn cung cấp. Stable Diffusion có thể được sử dụng cho nhiều mục đích khác nhau, bao gồm tạo ra nghệ thuật, minh họa ý tưởng, tạo ảnh cho các mục đích thương mại và nhiều hơn nữa. 

# Giới thiệu về Stable Diffusion
Trước tiên để bắt đầu cho việc tìm hiểu về Stable Diffusion, thì mình sẽ giới thiệu về mô hình sinh trước. Sau đó mình sẽ giới thiệu về động lực để các tác giả tạo ra mô hình này (fun fact ☝️🤓: stable diffusion là bản cải tiến từ mô hình diffusion có từ 2015) và sau đó là cơ chế hoạt động của mô hình này, cuối cùng sẽ là phần code để implement lại từ đầu mô hình này. 

# Các kiến thức liên quan 
## 1. Mô hình sinh (Generative model)

![genvsdis](/assets/img/paper%20explained%206/genvsdis.png)
_Discriminative versus Generative model_

Giới thiệu sơ qua thì Discriminative model là mô hình phân loại (hoặc là mô hình điều kiện (conditional models) như một vài nơi khác đặt tên), loại mô hình này sẽ học các đường biên quyết định (decision boundaries) để cho ra các đáp án là có hoặc không, đúng hoặc sai, thắng hoặc thua, vân vân. Ngoài ra loại mô hình này cũng có thể được phát triển lên để phân loại đa nhãn chứ không phải nhãn nhị phân như vừa ví dụ. Nhưng do chức năng của các Discriminative model là như vậy, nên loại mô hình này **không có khả năng tạo ra các các điểm dữ liệu mới**. 

Mô hình sinh (Generative model) đúng như tên gọi của nó thì lại cho phép chúng ta **tạo ra các điểm dữ liệu mới tuân theo phân phối xác suất tương tự như phân phối xác suất của dữ liệu mà chúng ta huấn luyện trên**. Từ đó mà mở ra nhiều hướng tiếp cận đa dạng (cho nhiều kiểu dữ liệu luôn), ví dụ như:
- Sinh ảnh (Image Synthesis)
- Tăng cường dữ liệu (Data Augmentation)
- Viết nhạc, làm thơ, v.v...
 
Lấy một ví dụ thực tế, bạn đang muốn tạo ra khoảng 100 sample về chiều cao của người châu á đi, và qua các dữ liệu bạn có được, bạn biết rằng chiều cao của người Châu Á tuân theo phân phối chuẩn $\mathcal{N}(160, 15^2)$ (số mình xạo đó, nhưng mà idea là vậy 🫥). Lúc này, các bạn sẽ đặt tên của biến ngẫu nhiên của các bạn là $X$, lúc này các bạn có thể biểu diễn dưới dạng toán học như sau: 

$$
\begin{equation}
X \sim \mathcal{N}(160, 15^2)
\end{equation}
$$

Thì cái mô hình sinh đơn giản là vậy, **một mô hình sinh sẽ học cái phân phối xác suất của tập dữ liệu, sau đó ta có thể lấy mẫu từ cái phân phối được học đó để tạo ra dữ liệu mới**. 

Về cơ bản thì nó sẽ là vậy, tuy nhiên các bạn nên tìm đọc sâu hơn để hiểu tường tận về vấn đề này. 

## 2. Mô hình khuếch tán (Diffusion Model)
Thường thì người ta chỉ nói là Diffusion Model thôi chứ không ai dùng từ mô hình khuếch tán cả (fun fact ☝️🤓: chữ Diffusion trong nhiệt động học có nghĩa là khuếch tán (và đây cũng là idea của bài này luôn)). 

![diffusion_schematics](/assets/img/paper%20explained%206/diffusion_schematics.png)
_Diffusion process_

Hình ở trên là tóm tắt toàn bộ ý tưởng của Diffusion Model, đó là một quá trình bao gồm 2 bước lớn:
- Quá trình khuếch tán thuận (forward process): Thêm nhiễu một cách **chậm rãi, có hệ thống**.
- Quá trình khuếch tán nghịch (backward process): **Đảo ngược** từ ảnh nhiễu về lại ảnh ban đầu (nghe vô lý nhưng thật ra có thể).

Về cụ thể 2 quá trình này, các bạn đợi mình ra bài mới nhé =)))), hoặc các bạn có thể đọc bài [này](https://viblo.asia/p/diffusion-models-co-ban-phan-1-E1XVOx884Mz) (Viết bằng tiếng việt, ngắn, gọn, dễ hiểu, thẳng vào vấn đề), hoặc đọc bài [này](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) (Viết bằng tiếng anh, dài, kĩ, người viết là lead của một team trong OpenAI). 

## 3. Motivation 
Một vấn đề cũng nằm ở trong hình ở trên, thể hiện điểm yếu của mô hình Diffusion truyền thống, đó là ở **kích thước của latent variable (biến tiềm ẩn) bằng đúng với kích thước của ma trận ảnh đầu vào**, mà một khi chúng ta lấy cái đó, mang đi cho vào mô hình khác để denoise, lấy ví dụ như U-Net chẳng hạn, chúng ta sẽ có kích thước ma trận rất lớn, dẫn đến hao tổn về chi phí tính toán. Đối với các ảnh đầu vào quá nhỏ, ví dụ như là MNIST với size ảnh là `28 x 28` thì không phải vấn đề, nhưng mọi thứ sẽ bét nhè nếu kích thước ảnh là `1024 x 1024`. 

Để giải quyết vấn đề đó, nhóm tác giả đã đề xuất một mô hình mới, mà giờ chúng ta gọi là Stable Diffusion (fun fact ☝️🤓: tên gốc của họ mô hình này là Latent Diffusion Model (LDM)). Dưới đây là quá trình mà mô hình này hoạt động: 

![ldm](/assets/img/paper%20explained%206/stable%20diffusion.png)
_Stable Diffusion Process_

## 4. CLIP
Trong hình trên, mọi người để ý sẽ thấy một layer có tên là `Text Encoder`, layer này sẽ làm nhiệm vụ chuyển văn bản đầu vào từ dạng ngôn ngữ tự nhiên sang dạng số để mô hình hiểu được (mô hình không đọc được chữ mọi người à, nó đọc được số thôi). Và trong bài báo gốc, các tác giả sử dụng mô hình `BERT` để làm cái layer này, nhưng phiên bản đầu của Stable Diffusion thì họ lại dùng `CLIP` của OpenAI. 

Mình sẽ lên một bài cụ thể về `CLIP` sau, nhưng mọi người có thể đọc từ chính tác giả của bài báo này để hiểu được nó làm gì, đây là bài [đó](https://openai.com/research/clip). Dưới đây là tổng quan mô hình này: 

![CLIP](/assets/img/paper%20explained%206/CLIP.png)
_CLIP Model_

CLIP (Contrastive Language–Image Pre-training) là một mô hình **cho chúng ta biết khả năng liên kết trực tiếp giữa văn bản và hình ảnh**, mở ra nhiều ứng dụng cho cả xử lý ngôn ngữ tự nhiên và thị giác máy tính. 

Mục tiêu khi OpenAI huấn luyện CLIP để nó trở thành một pre-train model đó là họ sẽ huấn luyện phần image encoder và text encoder để dự đoán xem các ảnh sẽ đi chung với câu nào trong bộ dữ liệu của họ. 

Với các lý do được nêu ở trên, CLIP không phải được thiết kế để giải quyết một vấn đề cụ thể nào cả, mô hình này có thể được sử dụng để giải quyết các bài toán ngách khác mà vẫn dể thở. 

## 5. Variational Autoencoder(VAE)

# Cách thức hoạt động

# Code 

# Thảo luận kết quả 

