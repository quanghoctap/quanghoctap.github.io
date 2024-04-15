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
# Các kiến thức liên quan 

# Cách thức hoạt động

# Code 

# Thảo luận kết quả 

