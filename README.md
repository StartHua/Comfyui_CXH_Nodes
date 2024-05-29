# Comfyui_CXH_Nodes

1.Net_image （网络图片加载兼容load image：开发过程遇到把图片上传到阿里云，等到执行队列时候又要把图片发送到要执行的机器，太麻烦）

![net_image](https://github.com/StartHua/Comfyui_CXH_Nodes/assets/22284244/c9151015-8689-4f52-a12e-63a117590ef7)

2.CLoad_Image 解决图片MPO 格式报错SyntaxError: not a JPEG file

![1716961041138](https://github.com/StartHua/Comfyui_CXH_Nodes/assets/22284244/296e6bea-3625-41d4-a98f-e7705020074a)


3.Phi-3-vision-128k-instruct 图片反推（自动下载模型到models\microsoft下）

![1716443218407](https://github.com/StartHua/Comfyui_CXH_Nodes/assets/22284244/b8241d4b-bf33-4849-a5c1-a059615d4e2b)

注意修改（主要是window端flash_attention 支持不太友好,模型自动下载失败请手动下载）查看模型下载后的README作者给出了修改方案：
phi-3-vision-128k-instruct-quantized 量化版需要flash_attention安装

![1716443127371](https://github.com/StartHua/Comfyui_CXH_Nodes/assets/22284244/752be09a-3022-4c25-ad2e-cc4d7c63183c)
