import { Schema, Prop, SchemaFactory } from '@nestjs/mongoose';
import { Document } from 'mongoose';

@Schema({ timestamps: true })
export class Student extends Document {
  @Prop({ required: true, unique: true })
  studentId: string;

  @Prop({ required: true })
  name: string;

  @Prop()
  enrollmentPhotoCount: number;

  @Prop()
  lastAttendanceDate?: Date;

  @Prop()
  totalAttendanceCount: number;
}

export const StudentSchema = SchemaFactory.createForClass(Student);
