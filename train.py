import torch
from torch.autograd import Variable
import time
from utils import AverageMeter, calculate_accuracy

def train_epoch_multimodal(epoch, data_loader, model, criterion, optimizer, opt, epoch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter() #changes

    end_time = time.time()
    for i, (audio_inputs, text_inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        targets = targets.to(opt.device)
        """
        input_ids = text_inputs['input_ids']
        attention_mask = text_inputs['attention_mask']
        token_type_ids = text_inputs['token_type_ids']

        print(audio_inputs.size())
        text_inputs = torch.stack((input_ids, attention_mask, token_type_ids), dim=1)
        print(text_inputs.size())
        """

        with torch.no_grad():
          coefficients = torch.randint(low=0, high=100, size=(audio_inputs.size(0), 1, 1)) / 100
          text_coefficients = 1 - coefficients  # Inverse for text
          coefficients = coefficients.repeat(1, audio_inputs.size(1), audio_inputs.size(2))
          text_coefficients = text_coefficients.repeat(1, text_inputs.size(1), text_inputs.size(2))
          audio_inputs = torch.cat( (audio_inputs, audio_inputs * coefficients, torch.zeros(audio_inputs.size()), audio_inputs),dim=0)
          text_inputs = torch.cat((text_inputs, text_inputs * text_coefficients, text_inputs, torch.zeros(text_inputs.size())),dim=0)
          targets = torch.cat((targets, targets, targets, targets), dim=0)
          shuffle = torch.randperm(audio_inputs.size(0))
          audio_inputs = audio_inputs[shuffle]
          text_inputs = text_inputs[shuffle]
          targets = targets[shuffle]



        audio_inputs = Variable(audio_inputs)
        text_inputs = Variable(text_inputs)

        #text_inputs = {'input_ids': text_inputs[:, 0, :].long(), 'attention_mask': text_inputs[:, 1, :].long(), 'token_type_ids': text_inputs[:, 2, :].long()}
        targets = Variable(targets)
        outputs = model(audio_inputs, text_inputs)
        loss = criterion(outputs, targets)

        losses.update(loss.data, audio_inputs.size(0))
        prec1, prec3 = calculate_accuracy(outputs.data, targets.data, topk=(1,3)) ##changes
        top1.update(prec1, audio_inputs.size(0))
        top3.update(prec3, audio_inputs.size(0)) #changes

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if i % 10 ==0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  'Prec@3 {top5.val:.5f} ({top5.avg:.5f})'.format(
                      epoch,
                      i,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      top5=top3, #changes
                      lr=optimizer.param_groups[0]['lr']))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'prec1': top1.avg.item(),
        'prec5': top3.avg.item(), #changes
        'lr': optimizer.param_groups[0]['lr']
    })
    train_losses.append(losses.avg.item())


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger):
    print('train at epoch {}'.format(epoch))

    if opt.model == 'multimodal':
        train_epoch_multimodal(epoch,  data_loader, model, criterion, optimizer, opt, epoch_logger)
        return
