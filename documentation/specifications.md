
## 🧠 Further Thoughts

 Let’s imagine what shiny new toys and big upgrades the little language model needs to turn into a grown-up large language model who knows all about the big wide world!

 **Known data sets**

| DataSet Type | DataSet Type | DataSet Name | DataSet Tokens |
|-----|-----|-----|-----|
| open | train | SlimPajama | 627B |
| open | train | RedPajama v1 | 1T |
| open | train | RedPajama v2 | 30T |
| open | eval | HellaSwag | 30T |

**Known language models**

| Model | Type | Parameters | Input Tokens | Output Tokens | Training Model Tokens | Training Model Flops | Training Environment | Training Environment Flops /s | Training Content | Training Duration |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| GPT2 | s | 117M | 1024 | Shared | 3.3B | 2.3e18F | 1-2 x A100 | 100P | WebText (Reddit outbound links with ≥3 karma; ~40GB of filtered internet text) | 60D |
| GPT2 | m | 335M | 1024 | Shared | 3.3B | 7e18F | 4-8 × A100 | 200P | Same as Small; byte-level BPE tokenization, 50,257 vocab size | 60D |
| GPT2 | l | 774B | 1024 | Shared | 3.3B | 15e18F | 8-16 × V100 | 400P | Same as Small; trained with causal LM objective | 60D |
| GPT2 | xl | 1.5B | 1024 | Shared | 3.3B | ~30e18F | 16-32 × V100 | 800P | Same as Small;  trained with causal LM objective | 60D |
| GPT3 | s | 125M | 2048 | Shared | 300B | 2.25e21F | 1-2 × A100 | 100P | Common Crawl (filtered), WebText2, Books1/2, Wikipedia (~570GB filtered) | 180D |
| GPT3 | m | 350M | 4096 | Shared | 300B | 6.3e21F | 8-16 × A100 | 200P | Same as Small; scaled architecture with 24 layers and 16 attention heads | 180D |
| GPT3 | l | 760M | 16384 | 4096 | 300B | 3.7e21F | 100-200 × A100 | 400P | Same as Small; deeper model with wider layers and more attention heads | 180D |
| GPT3 | xl | 6.7B | 2048 | Shared | 300B | ~1.2e22F | 32-64 × A100 | 800P | Common Crawl, WebText2, Books1/2, Wikipedia (~570GB filtered) | 180D |
| GPT4 | s | 1B | 8192 | 8192 | 6B | 1.8e21F | 100-200 × A100 | 1OOP | Filtered Common Crawl, Books, Wikipedia, WebText2, code, academic papers | 160D |
| GPT4 | m | 13B | 32768 | 8192 | 1.7T | 9.4e23F | 400-600 × A100 | 400P | Same as Small; with broader multilingual and multimodal data | 160D |
| GPT4 | l | 65B | 128000 | 4096 | 13T | 3e25F | 2k-4K × A100 | 1E | Massive curated dataset: text, code, images, audio (for GPT-4o), RLHF tuning | 90D  |
| LLAMA2 | s | 7B | 4096 | Shared | 2T | 1.5e24F | 32-64 × A100 | 400P | Publicly available web data (filtered), books, code, academic papers | 180D |
| LLAMA2 | m | 13B | 4096 | Shared | 2T | 2.6e24F | 128-256 × A100 | 400P | Same as Small; with additional curated datasets for scaling | 180D |
| LLAMA2 | l | 70B | 4096 | Shared | 2T | 14e24F | 1024K+ x A100 | 800P | Same as Small; plus enhanced filtering, grouped-query attention optimization | 180D |
| LLAMA3 | s | 8B | 8000 | Shared | 15T | 7.2e24F | 64-128 x A100 | 700P | Books, Wikipedia, GitHub, StackExchange | 70D |
| LLAMA3 | m | 70B | 128000 | Shared | 15T | 63e24F | 512-1024 x A100 | 800P | Books, Wikipedia, GitHub, StackExchange | 70D |
| LLAMA3 | l | 405B | 128000 | Shared | 15T | 365e24F | 1024+ x A100 | 1E | Books, Wikipedia, GitHub, StackExchange | 70D |
| LLAMA4 Scout | s | 109B total / 17B active | 10000000 | Shared | ~30T | ~8e25F | 32-64 x H100 |	~400T |	Text, image, video (multimodal) |	Unknown |
| LLAMA4 Maverick | m | 400B total / 17B active | 10000000 | Shared | ~30T | ~38e25F | 128-256 × H100 | ~3200T | Text, image, code, multilingual data | Unknown |
| LLAMA4 Maverick | l | 2T total / 288B active | 10000000 | Shared | ~30T | ~100e25F | 32K+ x H100 | Unknown | STEM-heavy, multimodal, synthetic distill. | Unknown |
| GPT-4o-nano | s | — | 128000 | 4096 | — | — | — | — | — | — |
| GPT-4o-mini | m | — | 128000 | 16096 | — | — | — | — | — | — |
| GPT-4o | l | — | 128000 | 4096 | — | — | — | — | — | — |
| GPT-4.1-nano | s | — | 1000000 | 32768 | — | — | — | — | — | — |
| GPT-4.1-mini | m | — | 1000000 | 32768 | — | — | — | — | — | — |
| GPT-4.1 | l | — | 1000000 | 32768  | — | — | — | — | — | — |
| o1-mini | m | — | 200000 | 100000 | — | — | — | — | — | — |
| o1 | l | — | 200000 | 100000 | — | — | — | — | — | — |
| o3-mini | s | — | 200000 | 100000 | — | — | — | — | — | — |
| o3 | m | — | 20000 0| 100000 | — | — | — | — | — | — |
| o3-pro | l | — | 200000 | 100000 | — | — | — | — | — | — |
| o4-mini | s | — | 200000 | 100000 | — | — | — | — | — | — |
| o4 | m | — | 200000 | 100000 | — | — | — | — | — | — |
| o4-pro | l | — | 200000 | 100000 | — | — | — | — | — | — |
| Grok-3 | — | — | 131072 | 16384 | — | — | — | — | — | — |
| Gemini 2.0 | — | — | 1048576| 8192 | — | — | — | — | — | — |
| Gemini 2.0 Flash | — | — | 1048576 | 8192 | — | — | — | — | — | — |
| Gemini 2.5 | — | — | 1048576 | 65535 | — | — | — | — | — | — |
| Gemini 2.5 Pro | — | — | 1048576 | 65535 | — | — | — | — | — | — |
| Claude Sonnet 3.5 | — | — | 200000 | 4096 | — | — | — | — | — | — |
| Claude Sonnet 3.7 | — | — | 200000 | 8192 | — | — | — | — | — | — |
| Claude Sonnet 4 | — | — | 200000 | 64000 | — | — | — | — | — | — |

*Do not try to relate Training Model Flops, Training Environment Training Environment Flops, Training Duration as there are other factors which are playing role, like: number of epochs, number of precision parallel efficiency, memory bandwidth, thermal limitations, etc.


** 🖥️ GPUs **

| CPU Model                 | Cores / Threads      | Base Speed      | Boost Speed    | Memory Bandwidth |
|---------------------------|----------------------|-----------------|----------------|------------------|
| Intel Core i7-1360P       | 12 (4P + 8E) / 16    | 2.2 GHz (P-core)| Up to 5.0 GHz  | 89.6 GB/s        |
| AMD EPYC Genoa 9654       | 96 / 192             | 2.4 GHz         | Up to 3.7 GHz  | 460 GB/s         |
| AMD EPYC 7763             | 64 / 128             | 2.45 GHz        | Up to 3.5 GHz  | 205 GB/s         |


** 🎨 GPUs **

| GPU Model                | Cores     | Core Speed              | Memory Type | Memory Size | Memory Speed              | Bandwidth   |
|--------------------------|-----------|-------------------------|-------------|-------------|---------------------------|-------------|
| RTX A500 Laptop GPU      | 2048      | 1440 MHz (1770 MHz)     | GDDR6       | 4 GB        | 1750 MHz (14 Gbps)        | 112 GB/s    |
| NVIDIA Tesla T4          | 2560      | 585 MHz (1590 MHz)      | GDDR6       | 16 GB       | 1250 MHz (10 Gbps)        | 320 GB/s    |
| NVIDIA Tesla V710 GPU    | 3456      | 1900 MHz (2000 MHz)     | GDDR6       | 28 GB       | 2250 MHz (18 Gbps)        | 504 GB/s    |
| NVIDIA Tesla P100        | 3584      | 1190 MHz (1329 MHz)     | HBM2        | 16 GB       | 715 MHz                   | 732 GB/s    |
| AMD Radeon Instinct MI25 | 4096      | 1400 MHz (1500 MHz)     | HBM2        | 16 GB       | 852 MHz                   | 436 GB/s    |
| NVIDIA Tesla M60         | 2048 ×2   | 557 MHz (1178 MHz)      | GDDR5       | 8 GB ×2     | 1253 MHz (5 Gbps)         | 256 GB/s ×2 |
| NVIDIA Tesla K80         | 2496 ×2   | 562 MHz (824 MHz)       | GDDR5       | 12 GB ×2    | 1253 MHz (5 Gbps)         | 480 GB/s    |
| NVIDIA Tesla V100        | 5120      | 1245 MHz (1380 MHz)     | HBM2        | 16 GB       | 876 MHz                   | 900 GB/s    |
| NVIDIA A10               | 9216      | 885 MHz (1695 MHz)      | GDDR6       | 24 GB       | 1563 MHz (12.5 Gbps)      | 600 GB/s    |