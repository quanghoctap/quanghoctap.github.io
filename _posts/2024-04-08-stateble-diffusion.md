---
title: 'Paper Explained 6: High Resolution Image Synthesis with Latent Diffusion Models'
date: 2027-04-08
categories: [Data Science, Deep Learning]
tags: [advanced, paper explained]     # TAG names should always be lowercase
toc: true
math: true
publish: true
---

Stable Diffusion là một mô hình trí tuệ nhân tạo (AI) tạo sinh được phát triển bởi Stability AI. Nó cho phép bạn tạo ra hình ảnh từ mô tả bằng văn bản. Stable Diffusion hoạt động dựa trên mô hình khuếch tán, bắt đầu với một bức tranh nhiễu và dần dần "khuếch tán" nhiễu đó để tạo ra hình ảnh mong muốn. Quá trình này được điều khiển bởi mô tả bằng văn bản mà bạn cung cấp. Stable Diffusion có thể được sử dụng cho nhiều mục đích khác nhau, bao gồm tạo ra nghệ thuật, minh họa ý tưởng, tạo ảnh cho các mục đích thương mại và nhiều hơn nữa. 

# Giới thiệu về Stable Diffusion
Trước tiên để bắt đầu cho việc tìm hiểu về Stable Diffusion, thì mình sẽ giới thiệu về mô hình sinh trước. Sau đó mình sẽ giới thiệu về động lực để các tác giả tạo ra mô hình này (fun fact ☝️🤓: stable diffusion là bản cải tiến từ mô hình diffusion có từ 2015) và sau đó là cơ chế hoạt động của mô hình này, cuối cùng sẽ là phần code để implement lại từ đầu mô hình này. 

Một vài ứng dụng của Stable Diffusion (mà có thể các bạn đã thấy hoặc chưa thấy):
![example](/assets/img/paper%20explained%206/example.png)
_Example for 1.45B Model with user input_

Và ngoài ra thì cũng có nhiều phiên bản cải tiến khác, xịn hơn nữa. 
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
Cũng trong hình mà mọi người thấy ở trên về cấu trúc của mô hình stable diffusion, nó sẽ có 2 phần đó là `Image Encoder` và `Image Decoder`. 

Quay trở lại với lí do mà nhóm tác giả viết bài này, có thể thấy rằng việc mà họ denoise trên một latent variable có kích thước bằng đúng với kích thước đầu vào sẽ làm cho việc denoise bức ảnh trở nên rất tốn kém, đặc biệt là khi mà kích thước bức ảnh lớn và bước timestep $T$ là rất lớn. Lúc này họ mới nghĩ tới việc **thay vì học phân phối $p(x)$ từ chính dữ liệu hình ảnh, thay vào đó, ta nên học phân phối của latent variable trên tập dữ liệu, sử dụng VAE**. 

![VAE](/assets/img/paper%20explained%206/VAE.jpg)
_VAE architecture_

Bằng cách áp dụng VAE để giảm số chiều dữ liệu xuống, chúng ta trực tiếp làm giảm kích thước ma trận để cho quá trình denoise, cụ thể là từ một bức ảnh `512 x 512` xuống thành một ma trận `64 x 64` .Vậy câu hỏi ở đây là: "*Tại sao VAE? 🤨 AE thường thì sao không được?*". 

Rõ ràng đối với objective của chúng ta, Autoencoder cũng là một lựa chọn hợp lí, tuy nhiên có lí do để chúng ta không sử dụng Autoencoder. Nguyên nhân chính yếu nhất để chúng ta không sử dụng Autoencoder đó là **biểu diễn trong không gian tiềm ẩn (latent space) không có mang ý nghĩa gì cả**. Mô hình của chúng ta sẽ chỉ gán cho cái ảnh đầu vào một vector input ngẫu nhiên (không hẳn là ngẫu nhiên, nhưng mà vấn đề là Autoencoder không có học được mối quan hệ có nghĩa nào giữa các điểm dữ liệu cả, trong chính cái objective của AE là nó cố gắng tái hiện lại input của nó mà, kiểu như mô hình này chỉ biết tới bản thân của nó thôi 😻👌). Dưới đây là kiến trúc của một mạng AE truyền thống: 

![AE](/assets/img/paper%20explained%206/autoencoder.png)
_AE architecture_

Điểm khác nhau của 2 mô hình VAE và AE nằm ở chỗ VAE thật sự cho phép chúng ta học ra một latent space có ý nghĩa.

## 6. Latent Space
Vậy câu hỏi kế tiếp (không quá liên quan đến bài này, nhưng biết thì cũng không hại gì): *"Latent Space là gì 😐?"*

Hiểu đơn giản, Latent Space (hay không gian ẩn) thường là một không gian có số chiều thấp hơn chiều của dữ liệu đầu vào và vẫn giữ được đặc tính của dữ liệu input đó, và chúng ta có thể làm được việc này thông qua các mô hình như VAE hay AE. 

![mnist_latent](/assets/img/paper%20explained%206/mnist-latent.png)
_Latent space of MNIST_

Như trong hình này là một latent space có ý nghĩa, các vector biểu diễn sau khi được nén vào trong latent space sẽ nằm gần nhau, ngụ ý rằng các sample gần giống nhau thì sẽ nằm gần nhau trong một không gian ẩn. Do đó khi chúng ta thực hiện lấy mẫu, nếu mà mẫu chúng ta lấy thuộc vào một cụm nào đó, thì khả năng rất cao chúng ta sẽ tái tạo lại được ảnh rất liên quan đến cụm đó. 

## 7. Denoising UNET 

Vai trò của UNET trong mô hình stable diffusion là rất quan trọng (và có vẻ như mô hình nào thuộc dạng Diffusion-based cũng sử dụng UNET để denoise). Có nhiều bài khác đã nói về UNET mà mọi người có thể tìm đọc(hoặc đợi mình lên bài 🥹). Lý do mà mô hình có tên UNET là bởi vì cấu trúc của nó tạo thành hình chữ U: 

![UNET](/assets/img/paper%20explained%206/UNET.png)
_UNET architecture_

Và ở một phương diện nào đó, đây cũng chính là một dạng Autoencoder! 

Nhưng mà nếu chỉ có vậy thì điều gì đã khiến UNET trở nên đặc biệt hơn? Câu trả lời nằm ở những đường màu xám ở hình trên. Những đường xám này chính là những kết nối tắc (Residual Connection). 

Những đường Residual Connection này, hiểu đơn giản là một cái cheatcode, vai trò của những đường kết nối này là hỗ trợ nửa sau của phần Decoder trong mạng UNET, cụ thể hơn là ở giai đoạn Encode, ở những bước đầu tiên, chúng ta thật sự có đầy đủ thông tin của bức ảnh đầu vào, có thể nói ở những bước này, mô hình phần nào đó bảo toàn những đặc trưng của ảnh. Nhưng vấn đề nằm ở nửa sau, phần Decode cố gắng phóng lớn lại cái ảnh đang được biểu diễn trong latent space, vấn đề là làm sao để phần Decode biết mình đi đúng hướng? Những cái kết nối tắc này sẽ giúp phần Decode biết là ở layer đó thì biểu diễn ma trận nên là như thế nào, ra sao. Từ đó nó không đi lạc nữa. 

Fun fact 🤓☝️: UNET ban đầu được thiết kế cho bài toán segmentation, nhưng do những tính chất mà mình vừa nêu, mọi người đã mang nó vô hầu hết các bài toán tạo ra ảnh cho các mô hình diffusion-base. 

## 8. Transformer 
Cũng đã có nhiều bài post viết chi tiết và cụ thể về mô hình Transformer rồi, mọi người có thể tìm đọc, trong khuôn khổ bài toán này, mình sẽ nói vắng tắt về cross-attention trong Transformer và cách nó được áp dụng trong bài này.

**Vai trò của cơ chế cross-attention**: Hiểu đơn giản, tưởng tượng bạn đang đọc một bức tranh, và đồng thời đọc luôn caption của bức tranh đó. Lúc này có thể bạn đang so sánh đối tượng trong tranh với các từ bạn quan sát được trong câu mô tả để xem nó có trùng không. 


# Cách thức hoạt động

Ok, sau khi đã có hiểu biết về các thành phần trong mô hình Stable Diffusion, giờ ta sẽ đi đến phần chính, đó là cấu trúc cụ thể của mô hình, mọi người có thể nhìn vào hình dưới đây: 

![stable_diffusion_full](/assets/img/paper%20explained%206/stable_diffusion_full.png)
_Full Component of Stable Diffusion Architecture_

Ở đây, có một điểm lưu ý đó là phần Encoder $\tau_\theta$. Ở trên mình có nói cụ thể cái đó là phần Text Encoder, nhưng trong paper gốc thì họ lại ghi là $\tau_\theta$, lý do cho việc này nằm ở chỗ phần Text thật ra là một trong những điều kiện giúp chúng ta điều chỉnh lại quá trình denoise. Nhưng ngoài thật ra, ngoài Text, chúng ta có thể thay đổi điều kiện đó thành những điều kiện khác sao cho phù hợp với task của chúng ta. Lấy ví dụ như với bài toán tô màu ảnh, mình hoàn toàn có thể thiết kế lại mô hình này sao cho nó nhận vào 2 bức ảnh, một bức ảnh đóng vai trò input, một bức ảnh đóng vai trò là điều kiện để điều chỉnh cho phù hợp với ý muốn của mình. **Có thể nói, Stable Diffusion bên cạnh giải quyết vấn đề về tính toán của Diffusion model, mô hình này còn cho phép đa dạng dữ liệu đầu vào, nhờ vào việc phần conditioning có thể đa dạng** 


# Code 

# Thảo luận kết quả 

