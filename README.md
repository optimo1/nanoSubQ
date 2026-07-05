# nanoSubQ AI

This is my journey of building **nanoSubQ AI** from scratch, using Andrej Karpathy's `nanoGPT` as a foundation. 

Since it’s been a while since I deeply touched Python and PyTorch, I'm spending the first few weeks mastering the basics again before hacking the model architecture.

---

## 💡 The Core Concept

The main goal of this project is to fix how modern AI handles long text:

* **Normal AI Architecture:** Think of it like a person reading a book who has to re-read every single previous word every time they encounter a new one. It gets exponentially slower and takes massive amounts of memory as the text gets longer.
* **SubQ AI Architecture:** Instead of looking at everything, the model uses a smart internal filter to instantly focus only on the most important, high-signal words from its history. This keeps the speed fast and steady no matter how long the text is.

---

## 📈 Follow My Journey

I am building this project completely in public and sharing all my milestones, breakthroughs, and code updates over on my LinkedIn:

👉 [www.linkedin.com/in/adilzhanturaliev](https://www.linkedin.com/in/adilzhanturaliev)

Connect and follow along to watch it come to life! 🚀



@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@#+@@@@@@@@    @@=@@@@@@@@@@@@@@@@     @@@@@@@@@@@@@@@@@     #@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@ @=  @@@@:  @@@@@@@@@@@@@@@@@@   @@@@@@@@@@@@@-   @@@@@    @@@  @@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@  @@    @@@@@@ @@@@@@@@@@@@@@@@@@   @@@@@@@@@@@@   @@@@@@@@@     @@@ @@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@ @-    @@@@@@@@    @@@    @@@@         @@@@@   @@@@@@@@@@@   @  @@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@  @@@@@@*       #@@@@@   @@@@@  @@@@   @@@@@   @@    @@@@@@@@@@@    @@@@@@@@  @@@@@@@@@@
@@@@=@@@@@@@@@@@@@@@@@.        @@@   @@@@@  @@@@   @@@@@@   @=   @@@@@@@@@@@    @@@@..@@@@@@@@@. @@@
@@@@@@@@@@@@@@@@@@@@ @@@@@@@   @@@   @@@@@  @@@@   @@@@@@   @@   @@@@@@@@@@@      @@@@@@@@@@@@@@@@@@
@@@@@@@@@  @@@@@..   =@@@@@@@  @@@   @@@@.  @@@@   @@@@@=  @@@@   @@@@@@@@@    @@* @@@@@@@ @@@@@@@@@
@@@@@@@@@@@@@= @@@@      +    @@@@@          @@@     +    @@@@@@%   @@@@@    @@@@@@@@  @@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    @@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@         @@@@  @@@@@@ @@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@       @@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@