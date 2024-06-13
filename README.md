<!--
 * @Author: hibana2077 hibana2077@gmail.com
 * @Date: 2024-06-06 14:50:01
 * @LastEditors: hibana2077 hibana2077@gmaill.com
 * @LastEditTime: 2024-06-13 15:07:09
 * @FilePath: \Dataset-Cartography-for-Tree-based-model\README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->

# Dataset-Cartography-for-Tree-based-model

MLPR 2024

## Usage

1. Clone the repository

```bash
git clone https://github.com/hibana2077Dataset-Cartography-for-Tree-based-model.git
```

2. Run docker

```bash
sudo docker run -v ./:/workspace --cpus="32" --name lab -dt ubuntu:22.04
```

3. Install dependencies

```bash
apt update && apt install -y python3 python3-pip
cd /workspace/Dataset-Cartography-for-Tree-based-model/src/main
pip3 install -r requirements.txt
```