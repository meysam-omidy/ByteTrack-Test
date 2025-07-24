import os
import numpy as np
from byte_tracker import BYTETracker
from evaluate import evaluate

class Args:
    def __init__(self):
        self.track_thresh = 0.5
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.mot20 = False
        self.min_box_area = 10
        self.aspect_ratio_thresh = 1.6


def run():
    seqs = ['MOT17-02-FRCNN']

    os.makedirs('outputs/bytetrack', exist_ok=True)

    seqmap = open('./trackeval/seqmap/mot17/custom.txt', 'w')
    seqmap.write('name\n')
    for seq in seqs:
        
        seqmap.write(f'{seq}\n')
        detections = np.loadtxt(f'detections/bytetrack_x_mot17/{seq}.txt', delimiter=',')
        gt_dets_file = np.loadtxt(f'MOT17/train/{seq}/gt/gt.txt', delimiter=',')

        args = Args()
        tracker = BYTETracker(args)
        results = []

        for i,frame_number in enumerate(np.unique(gt_dets_file[:,0])):
            dets = detections[detections[:, 0] == frame_number][:, 1:]
            online_targets = tracker.update(dets)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{frame_number},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )

        with open(f"outputs/bytetrack/{seq}.txt", 'w') as f:
            f.writelines(results)
    seqmap.close()


if __name__ == '__main__':
    print('tracking...')
    run()
    print('evaluating...')
    evaluate()