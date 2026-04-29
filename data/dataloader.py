"""
Working bAbI Dataset Loader
Compatible with Python 3.10+
"""

import tarfile
import urllib.request
from pathlib import Path


class BabiDataset:
    MIRRORS = [
        "https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz",
        "https://dl.fbaipublicfiles.com/babi/babiTasks1-20_v1-2.tar.gz"
    ]

    TASK_DESCRIPTIONS = {
        1:"1 supporting fact",
        2:"2 supporting facts",
        3:"3 supporting facts",
        4:"2 argument relations",
        5:"3 argument relations",
        6:"yes/no questions",
        7:"counting",
        8:"lists/sets",
        9:"simple negation",
        10:"indefinite knowledge",
        11:"basic coreference",
        12:"conjunction",
        13:"compound coreference",
        14:"time reasoning",
        15:"basic deduction",
        16:"basic induction",
        17:"positional reasoning",
        18:"size reasoning",
        19:"path finding",
        20:"agent motivation",
    }

    def __init__(self, data_dir="./data", download=True):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "babi_raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        if download and not self._exists():
            self.download()

    def _exists(self):
        return (self.raw_dir / "tasks_1-20_v1-2").exists()

    def download(self):
        print("Downloading bAbI dataset...")

        tar_path = self.raw_dir / "babi.tar.gz"

        if not tar_path.exists():
            downloaded = False

            for url in self.MIRRORS:
                try:
                    print(f"Trying {url}")

                    req = urllib.request.Request(
                        url,
                        headers={"User-Agent":"Mozilla/5.0"}
                    )

                    with urllib.request.urlopen(req) as r:
                        with open(tar_path, "wb") as f:
                            f.write(r.read())

                    print("Download successful.")
                    downloaded = True
                    break

                except Exception as e:
                    print(f"Failed: {e}")

            if not downloaded:
                raise RuntimeError("All download mirrors failed.")

        print("Extracting...")

        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(self.raw_dir)

        print("Extraction complete.\n")

    def load_task(
        self,
        task_id,
        train=True,
        max_size=None
    ):
        if task_id not in range(1,21):
            raise ValueError("task_id must be 1-20")

        split = "train" if train else "test"

        task_dir = self.raw_dir / "tasks_1-20_v1-2" / "en"

        matches = list(task_dir.glob(f"qa{task_id}_*_{split}.txt"))

        if not matches:
            raise FileNotFoundError(
                f"No file found for task {task_id}"
            )

        file_path = matches[0]

        print(
            f"Loading Task {task_id}: "
            f"{self.TASK_DESCRIPTIONS[task_id]}"
        )

        return self.parse_babi(file_path, max_size)

    def load_all_tasks(
        self,
        train=True,
        max_size_per_task=100
    ):
        tasks = {}

        for i in range(1,21):
            try:
                stories, questions, vocab = self.load_task(
                    i,
                    train=train,
                    max_size=max_size_per_task
                )

                tasks[i] = {
                    "stories":stories,
                    "questions":questions,
                    "word2idx":vocab,
                    "description":
                        self.TASK_DESCRIPTIONS[i]
                }

                print()

            except Exception as e:
                print(
                    f"Task {i} failed: {e}"
                )

        return tasks

    def load_all_tasks_joint(self, train=True, max_size_per_task=None):
        """Load all 20 tasks merged into a single shared-vocabulary corpus."""
        vocab = {"<pad>": 0, "<unk>": 1}
        next_idx = 2
        all_stories, all_questions = [], []

        for task_id in range(1, 21):
            stories, questions, _ = self.load_task(
                task_id, train=train, max_size=max_size_per_task
            )
            story_offset = len(all_stories)

            for s in stories:
                for sentence in s["sentences"]:
                    for w in self.tokenize(sentence):
                        if w not in vocab:
                            vocab[w] = next_idx
                            next_idx += 1

            for q in questions:
                for w in self.tokenize(q["question"]) + [q["answer"].lower()]:
                    if w not in vocab:
                        vocab[w] = next_idx
                        next_idx += 1
                q["story_id"] += story_offset
                q["task_id"] = task_id

            all_stories.extend(stories)
            all_questions.extend(questions)

        print(f"Joint corpus: {len(all_questions)} questions, vocab={len(vocab)}")
        return all_stories, all_questions, vocab

    @staticmethod
    def tokenize(sentence):
        return (
            sentence.lower()
            .replace(".","")
            .replace("?","")
            .split()
        )

    @staticmethod
    def parse_babi(filepath,max_size=None):

        stories=[]
        questions=[]

        vocab={
            "<pad>":0,
            "<unk>":1
        }

        next_idx=2

        current_story=[]

        with open(
            filepath,
            encoding="utf-8"
        ) as f:

            for line in f:

                line=line.strip()

                if not line:
                    continue

                parts=line.split("\t")

                # split "1 Mary moved ..."
                nid,text=parts[0].split(" ",1)
                nid=int(nid)

                # new story begins
                if nid==1:
                    current_story=[]

                if len(parts)==1:
                    # sentence
                    current_story.append(text)

                    for w in BabiDataset.tokenize(text):
                        if w not in vocab:
                            vocab[w]=next_idx
                            next_idx+=1

                else:
                    # question line
                    question=text
                    answer=parts[1]
                    support=list(
                        map(int,parts[2].split())
                    )

                    stories.append({
                        "sentences":
                            current_story.copy(),
                        "supporting_facts":
                            support
                    })

                    questions.append({
                        "question":
                            question,
                        "answer":
                            answer,
                        "supporting":
                            support,
                        "story_id":
                            len(stories)-1
                    })

                    for w in (
                        BabiDataset.tokenize(
                            question
                        )+[answer.lower()]
                    ):
                        if w not in vocab:
                            vocab[w]=next_idx
                            next_idx+=1

                if (
                    max_size and
                    len(questions)>=max_size
                ):
                    break

        print(
            f"{len(questions)} examples loaded"
        )
        print(
            f"Vocabulary size: {len(vocab)}"
        )

        return (
            stories,
            questions,
            vocab
        )

    @staticmethod
    def print_stats(tasks):

        print("\nDATASET STATS")
        print("="*60)

        for tid,data in tasks.items():

            avg_story=(
                sum(
                    len(
                        s["sentences"]
                    )
                    for s in data["stories"]
                )
                /
                len(data["stories"])
            )

            print(
                f"Task {tid:2d}: "
                f"{data['description']:25s}"
            )

            print(
                f"  Examples:"
                f"{len(data['questions'])}"
            )

            print(
                f"  Vocabulary:"
                f"{len(data['word2idx'])}"
            )

            print(
                f"  Avg story:"
                f"{avg_story:.1f}"
            )

            print()


def main():

    dataset=BabiDataset()

    print(
        "Example: Task 1\n"
    )

    stories,questions,vocab=dataset.load_task(
        1,
        max_size=5
    )

    if questions:

        q=questions[0]

        print(
            "\nSample Question:"
        )
        print(
            q["question"]
        )

        print(
            "Answer:",
            q["answer"]
        )

        print(
            "\nStory:"
        )

        for i,s in enumerate(
            stories[
                q["story_id"]
            ]["sentences"],
            1
        ):
            print(
                f"{i}. {s}"
            )

        print(
            "\nSupporting facts:",
            q["supporting"]
        )

    print(
        "\nLoading all tasks...\n"
    )

    tasks=dataset.load_all_tasks(
        max_size_per_task=100
    )

    dataset.print_stats(
        tasks
    )

    print(
        "\nDone."
    )


if __name__=="__main__":
    main()