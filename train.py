import torch


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x_img, x_img_clip, x_text, x_text_clip, x_vec, x_score, y) in enumerate(dataloader):
        x_img, x_text, y = x_img.to(device), x_text.to(device), y.to(device)
        x_img_clip, x_text_clip = x_img_clip.to(device), x_text_clip.to(device)
        x_vec, x_score = x_vec.to(device), x_score.to(device)
        pred = model(x_img, x_img_clip, x_text, x_text_clip, x_vec, x_score)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 30 == 0:
            loss, current = loss.item(), batch * len(x_img)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device, is_train=True):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct, correct_top3 = 0, 0, 0
    with torch.no_grad():
        for batch, (x_img, x_img_clip, x_text, x_text_clip, x_vec, x_score, y) in enumerate(dataloader):
            x_img, x_text, y = x_img.to(device), x_text.to(device), y.to(device)
            x_img_clip, x_text_clip = x_img_clip.to(device), x_text_clip.to(device)
            x_vec, x_score = x_vec.to(device), x_score.to(device)
            pred = model(x_img, x_img_clip, x_text, x_text_clip, x_vec, x_score)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            top3_pred = torch.topk(pred, 3).indices
            correct_top3 += torch.sum(top3_pred == y.unsqueeze(1)).item()

    test_loss /= num_batches
    correct /= size
    correct_top3 /= size

    flag = "Train" if is_train else "Test"
    print(
        f"{flag} Error: \n Top1 Acc: {(100 * correct):>0.1f}%, Top3 Acc: {(100 * correct_top3):>0.1f}%, Avg loss: {test_loss:>8f} ")
    return test_loss, correct, correct_top3
