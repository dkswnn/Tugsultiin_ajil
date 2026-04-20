import { Schema, Prop, SchemaFactory } from '@nestjs/mongoose';
import { Document } from 'mongoose';

@Schema({ timestamps: true })
export class AttendanceSession extends Document {
  @Prop({ required: true })
  sessionId: string;

  @Prop({ required: true })
  timestamp: Date;

  @Prop({ required: true })
  filename: string;

  @Prop({ required: true })
  faceIndex: number;

  @Prop({ required: true })
  studentId: string;

  @Prop({ required: true })
  score: number;

  @Prop()
  annotatedImageUrl?: string;

  @Prop()
  isPredictionCorrect?: boolean; // true=TP, false=FP/FN, undefined=not manually verified

  @Prop()
  notes?: string;
}

export const AttendanceSessionSchema =
  SchemaFactory.createForClass(AttendanceSession);
