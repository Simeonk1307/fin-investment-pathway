# Real-Time SEC Filing Analyzer

This project is a high-performance streaming pipeline that monitors the **US Securities and Exchange Commission (SEC)** in real-time.

It detects new financial filings (like **10-K Annual Reports**, **Insider Trading Form 4**, and **Whale Investor 13G**) and instantly processes them to extract valuable insights using **Apache Kafka** and **Pathway**.

---

##  Prerequisites
**System Requirements:**
* **OS:** Debian or Linux (Recommended)
* **Java:** JDK 11 or higher (Required for Kafka)
* **Python:** 3.10 or higher

---

## Part 1: Installation & Setup
*Do this once to set up your environment.*

### Install System Dependencies
Open a terminal and run:

sudo apt update
sudo apt install default-jre python3-pip python3-venv wget -y


cd ~
# Download Kafka 3.9.0 (Stable) to your home directory
wget [https://downloads.apache.org/kafka/3.9.0/kafka_2.13-3.9.0.tgz](https://downloads.apache.org/kafka/3.9.0/kafka_2.13-3.9.0.tgz)

# Extract and rename folder for easier access
tar -xzf kafka_2.13-3.9.0.tgz
mv kafka_2.13-3.9.0 kafka

# Cleanup zip file
rm kafka_2.13-3.9.0.tgz

# Create project directory
mkdir -p ~/sec_project
cd ~/sec_project

# Create Virtual Environment (venv)
python3 -m venv venv

# Activate venv
source venv/bin/activate

# Install Libraries
pip install pathway kafka-python feedparser requests beautifulsoup4

## Part 2: How to Run the Pipeline

you will need 4 terminals for this coming process

### TERMINAL 1: Zookeeper (The Manager)

cd ~/kafka
bin/zookeeper-server-start.sh config/zookeeper.properties

### TERMINAL 2: Kafka Broker (The Engine)

cd ~/kafka
bin/kafka-server-start.sh config/server.properties

### TERMINAL 3: The Producer (The Fetcher)

cd ~/sec_project
source venv/bin/activate

python producer.py

### TERMINAL 4: The Consumer (The Processor)

cd ~/sec_project
source venv/bin/activate

python consumer.py

## Part 3: Viewing the Output

cd ~/sec_project
tail -f output_smart.jsonl

```bash